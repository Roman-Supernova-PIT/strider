"""The STRIDER network: how well an object's spectrum matches each class at each candidate redshift.

The coadd term matches the coadd with the reference spectra, the spectral model averaged over each of five phase bins.
It compares learned token encodings of the coadd and of each reference spectrum over their common support, in the
rest frame of each candidate redshift, in two views: the spectrum itself and its continuum-removed form.

The source detector says, from summaries of the flux, whether the visits hold a transient at all
(`strider.network.source_detector`).
"""

import numpy as np
import torch
from torch import nn

from ..posterior import redshift_grid
from .coadd import edge_taper, inverse_variance_coadd, relative_inverse_variance
from .encoder import ContinuumRemoval, SpectralEncoder, scale_free_normalise, velocity_to_log_bins
from .matching import (
    interpolate,
    interpolate_error,
    interpolation,
    inverse_variance,
    log_mean_exp,
    normalisation,
    pool_weight,
    token_cosine,
)
from .source_detector import SourceDetector
from .spectral_model import SpectralModel, reference_spectra

__all__ = ["Network", "WEIGHTS"]

# encoding-grid points per rest-frame pixel: front features are computed once on this grid, then interpolated
# onto each candidate redshift's rest-frame grid
ENCODING_POINTS = 4
VIEWS = ("full", "continuum_removed")
TABLE = ("lower", "upper", "weight", "valid")
# each part's prefix in weights.pt, under which a module's tensors keep their names
WEIGHTS = {
    "spectral_model": "roman_reference.spectral_model",
    "spectral_encoder": "roman_reference.spectral_encoder",
    "continuum_removed_intercept": "roman_reference.continuum_removed_intercept",
    "continuum_removed_redshift_slope": "roman_reference.continuum_removed_redshift_slope",
    "log_coadd_scale": "roman_reference.log_coadd_scale",
    "log_prototype_temperature": "roman_reference.log_prototype_temperature",
    "phase_nodes_days": "roman_reference.phase_nodes_days",
    "window_weight": "roman_reference.window_weight",
    "coadd_profile_mask": "roman_reference.coadd_profile_mask",
    "coadd_profile_supported": "roman_reference.coadd_profile_supported",
    "source_detector": "evidence_sufficiency",
}


class Network(nn.Module):
    """The network of a model package.

    Parameters
    ----------
    config : dict
        The declared model config (``model_config.json``).
    spectral_model : dict
        The contents of ``spectral_model.pt``.
    chunk_size : int
        Candidate redshifts scored at once. It bounds memory and does not change the result.
    wavelength_grid : numpy.ndarray, optional
        The observed wavelength grid, ``wavelength_grid.npy``; by default, the one ``config`` describes.
    """

    def __init__(self, config, spectral_model, chunk_size=40, wavelength_grid=None):
        super().__init__()
        model, self.settings = config["model"], config["reference"]
        observation = config["observation"]
        observed = (
            np.geomspace(observation["wavelength_min"], observation["wavelength_max"], observation["wavelength_bins"])
            if wavelength_grid is None
            else np.asarray(wavelength_grid, dtype=np.float64)
        )
        rest = spectral_model["rest_wavelength"].numpy().astype(np.float64)
        redshift = redshift_grid(model)
        self.classes = list(model["classes"])
        if list(spectral_model["class_names"]) != self.classes:
            raise ValueError(
                f"spectral_model.pt has the classes {list(spectral_model['class_names'])} and "
                f"model_config.json {self.classes}; this STRIDER needs them equal. Upgrade "
                "STRIDER (pip install -U roman-snpit-strider), or use a model package made for this "
                "version"
            )
        self.chunk_size = chunk_size

        self.spectral_model = SpectralModel(
            spectral_model["rest_wavelength"].numpy(),
            len(self.classes),
            spectral_model["phase_range_days"],
            **spectral_model["settings"],
        )
        self.spectral_encoder = SpectralEncoder(
            self.settings["token_dim"],
            self.settings["attention_heads"],
            model["dropout"],
            self.settings["token_pool_size"],
            self.settings["minimum_encoder_support"],
            self.settings["encoder_feedforward_multiplier"],
        )
        width = self.settings["continuum_width_km_s"]
        self.continuum_removal = ContinuumRemoval(velocity_to_log_bins(observed[0], observed[-1], len(observed), width))
        self.rest_continuum_removal = ContinuumRemoval(velocity_to_log_bins(rest[0], rest[-1], len(rest), width))
        # the continuum-removed view's share by redshift, the coadd term's weight, the phase-bin temperature
        for name in (
            "continuum_removed_intercept",
            "continuum_removed_redshift_slope",
            "log_coadd_scale",
            "log_prototype_temperature",
        ):
            self.register_parameter(name, nn.Parameter(torch.zeros(())))
        counted = model.get("evidence_use_visit_count_and_span", True)
        self.source_detector = SourceDetector(
            model["dropout"], model.get("evidence_visit_count_reference", 12) if counted else None
        )

        self._buffer("redshift", torch.from_numpy(redshift))
        redshift = redshift.astype(np.float64)
        coordinate = (redshift - np.median(redshift)) / max(float(np.ptp(redshift)), 1.0)
        self._buffer("redshift_coordinate", torch.from_numpy(coordinate.astype(np.float32)))
        self._init_grids(observed, rest, redshift)
        # where the reference spectra are used, set from weights.pt by `load_weights`
        classes, nodes, pixels = spectral_model["coverage_counts"].shape
        bins = len(self.settings["phase_edges_days"]) - 1
        self.register_buffer("phase_nodes_days", torch.zeros(nodes))
        self.register_buffer("window_weight", torch.zeros(classes, bins, nodes))
        self.register_buffer("coadd_profile_mask", torch.zeros(classes, bins, pixels, dtype=torch.bool))
        self.register_buffer("coadd_profile_supported", torch.zeros(classes, bins, dtype=torch.bool))
        self.references = self.reference_spectra = None

    def _buffer(self, name, value):
        self.register_buffer(name, value, persistent=False)

    def _init_grids(self, observed, rest, redshift):
        """The fixed weights of the observed pixels, and the interpolation tables between the grids: observed to
        encoding grid, observed to each redshift's rest-frame grid, and encoding grid to each rest-frame grid."""
        trim = self.settings["edge_trim_fraction"]
        position = (np.log(observed) - np.log(observed[0])) / np.log(observed[-1] / observed[0])
        inside = torch.from_numpy(((position >= trim) & (position <= 1.0 - trim)).astype(np.float32))
        weight = edge_taper(len(observed), self.settings["edge_taper_fraction"]) * inside
        self._buffer("matching_weight", weight)
        self._buffer("matching_mask", (weight > 0.0).float())
        step = np.log(rest[-1] / rest[0]) / (len(rest) - 1) / ENCODING_POINTS
        count = int(np.floor(np.log(observed[-1] / observed[0]) / step - 0.5)) + 1
        encoding = observed[0] * np.exp((np.arange(-(-count // ENCODING_POINTS) * ENCODING_POINTS) + 0.5) * step)
        rest_frame = rest[None, :] * (1.0 + redshift[:, None])
        for name, source, target in (
            ("encoding", observed, encoding),
            ("rest_frame", observed, rest_frame),
            ("shift", encoding, rest_frame),
        ):
            for part, value in zip(TABLE, interpolation(source, target)):
                self._buffer(f"{name}_table_{part}", value)
        edge = interpolate(self.matching_weight, self.matching_mask, self._table("rest_frame"))[0]
        self._buffer("rest_frame_edge_weight", edge)

    def _table(self, name, start=0, stop=None):
        return tuple(getattr(self, f"{name}_table_{part}")[start:stop] for part in TABLE)

    def weight_names(self):
        """The name in weights.pt of each tensor of the network, by `WEIGHTS`."""
        names = {}
        for key in self.state_dict():
            part = next(name for name in WEIGHTS if key == name or key.startswith(f"{name}."))
            names[key] = WEIGHTS[part] + key.removeprefix(part)
        return names

    def load_weights(self, weights):
        """Load a package's ``weights.pt``, each tensor by `weight_names` and at its exact shape.

        Besides the parameters, it records the spectral model's phase nodes, the phase-bin weights and the masks of
        the reference spectra. It must hold nothing else.
        """
        names = self.weight_names()
        unknown = sorted(weights.keys() - set(names.values()))
        missing = sorted(set(names.values()) - weights.keys())
        if unknown or missing:
            raise ValueError(
                f"weights.pt does not fit this STRIDER: unknown {unknown}, missing {missing}. Upgrade "
                "STRIDER (pip install -U roman-snpit-strider), or use a model package made for this "
                "version"
            )
        state = {key: weights[name] for key, name in names.items()}
        wrong = [names[key] for key, value in self.state_dict().items() if state[key].shape != value.shape]
        if wrong:
            raise ValueError(
                f"weights.pt holds {', '.join(wrong)} at another shape than model_config.json gives; "
                "the package is damaged. Download it again"
            )
        self.load_state_dict(state)

    @torch.no_grad()
    def cache_references(self):
        """Compute the reference spectra and encode them, once: they depend on no input. Returns the reference
        spectra."""
        spectra = reference_spectra(
            self.spectral_model,
            self.phase_nodes_days,
            self.window_weight,
            self.coadd_profile_mask,
            self.rest_continuum_removal,
        )
        pixels = self.coadd_profile_mask.shape[-1]
        mask = self.coadd_profile_mask.reshape(-1, pixels)
        self.references = {}
        for name, values in spectra.items():
            tokens, support = self.spectral_encoder(scale_free_normalise(values.reshape(-1, pixels), mask), mask)
            self.references[name] = (tokens, support, self.coadd_profile_supported.reshape(-1), values.shape[:-1])
        self.reference_spectra = spectra
        return spectra

    def forward(self, flux, mask, log_error, scale, days):
        """The scores of one object.

        Parameters
        ----------
        flux, mask, log_error : torch.Tensor
            (visits, pixels) in time order, as `strider.preprocess.prepare` gives them.
        scale, days : torch.Tensor
            (visits,) visit scales and observer-frame days.

        Returns
        -------
        dict of torch.Tensor
            ``coadd_term`` (classes, z), the coadd's score after its learned weight, and ``support``, where a coadd
            reference can score it; ``source_logit``, the source detector's output.
        """
        return {key: value[0] for key, value in self.forward_many([(flux, mask, log_error, scale, days)]).items()}

    def forward_many(self, objects):
        """`forward` for several objects at once, each a tuple of its inputs as `forward` takes them, every output
        with a leading object axis. They are scored together, each with its own visits; an object's outputs do not
        depend on the others or on how many there are, up to rounding."""
        # each coadd and its view are made alone, where batching changes the rounding most; the costly scores at
        # every redshift are made together
        views = []
        for flux, mask, log_error, scale, _ in objects:
            coadd_flux, coadd_error, coadd_mask = inverse_variance_coadd(flux, mask, log_error, scale)
            views.append(self.matching_view(coadd_flux, coadd_mask, coadd_error))
        view = {key: torch.stack([v[key] for v in views]) for key in views[0]}
        coadd_score, support = self.coadd_route(view)
        return {
            "coadd_term": self.log_coadd_scale.exp() * coadd_score,
            "support": support,
            "source_logit": torch.stack([self.source_detector(o[0], o[1], o[4]) for o in objects]),
        }

    @torch.no_grad()
    def phase_bin_scores(self, view, index):
        """The coadd's score against each class's reference spectrum of each phase bin at the redshift ``index``,
        (classes, phase bins), its two views combined as `coadd_fine` combines them, and whether it is valid."""
        (full, full_valid), (removed, removed_valid) = self.scores(view, index, index + 1)
        share = torch.sigmoid(
            self.continuum_removed_intercept + self.continuum_removed_redshift_slope * self.redshift_coordinate[index]
        )
        return ((1.0 - share) * full + share * removed)[0], (full_valid & removed_valid)[0]

    def coadd_route(self, view):
        """The coadd's score and its support, (..., classes, z): its best phase bins, pooled per class."""
        scores, supports = zip(*(self.coadd_fine(view, *chunk) for chunk in self._chunks()))
        score, support = torch.cat(scores, -2).transpose(-1, -2), torch.cat(supports, -2).transpose(-1, -2)
        # one subtype per class, so T log mean exp(v / T) is v up to rounding; kept so the logits match the trained
        # network's to the last bit
        temperature = self.settings["fine_class_temperature"]
        return torch.where(support, temperature * (score / temperature), 0.0), support

    def coadd_fine(self, view, start, stop, share):
        """The coadd's score for each class at redshifts start:stop, (z, classes), and its support."""
        temperature = self.log_prototype_temperature.exp()
        (full, full_support), (removed, removed_support) = (
            log_mean_exp(score, valid, temperature) for score, valid in self.scores(view, start, stop)
        )
        share = share[:, None]
        return (1.0 - share) * full + share * removed, full_support & removed_support

    def _chunks(self):
        """Chunks of candidate redshifts, with the continuum-removed view's share at each."""
        share = torch.sigmoid(
            self.continuum_removed_intercept + self.continuum_removed_redshift_slope * self.redshift_coordinate
        )
        for start in range(0, len(self.redshift), self.chunk_size):
            stop = min(start + self.chunk_size, len(self.redshift))
            yield start, stop, share[start:stop]

    def matching_support(self, mask, error):
        """The pixels used in matching, those with weight and above a numerical precision floor, and each pixel's
        reliability: its relative inverse variance times its weight."""
        mask = mask * self.matching_mask
        reliability = relative_inverse_variance(error, mask)
        floor = self.settings["minimum_relative_spectral_precision"]
        mask = mask * (mask.bool() & torch.isfinite(reliability) & (reliability >= floor)).to(mask.dtype)
        return mask, reliability * self.matching_weight

    def matching_view(self, flux, mask, error):
        """Spectra on the observed grid (..., pixels), prepared for matching.

        The mask is `matching_support`'s. Each view (the spectrum and its continuum-removed form) is standardised
        once, with constants over the whole spectrum, and the encoder's front runs once, on the encoding grid.

        Returns
        -------
        dict of torch.Tensor
            ``mask``, ``reliability`` and ``error`` (in units of the spectrum's median error) on the observed grid,
            and each view's front features (..., G, d) on the encoding grid.
        """
        mask, reliability = self.matching_support(mask, error)
        removed = self.continuum_removal(flux, mask, reliability)

        table = self._table("encoding")
        edge = interpolate(self.matching_weight, self.matching_mask, table)[0]
        grid_mask = interpolate(mask, mask, table)[1]
        sampled = mask.bool() & grid_mask.bool().any(dim=-1, keepdim=True)
        grid_error = interpolate_error(error, mask, table)
        valid = grid_mask.bool() & torch.isfinite(grid_error) & (grid_error > 0.0)
        median = torch.nanmedian(torch.where(valid, grid_error.log(), torch.nan), dim=-1, keepdim=True).values
        error = torch.where(torch.isfinite(median), error / median.exp(), error)
        weight = inverse_variance(interpolate_error(error, mask, table), grid_mask) * edge
        view = {"mask": mask, "reliability": reliability, "error": error, "removed": removed}
        for name, values in zip(VIEWS, (flux, removed)):
            scale, mean, deviation = normalisation(interpolate(values, mask, table)[0], grid_mask, weight)
            values = torch.where(sampled, (values / scale - mean) / deviation, 0.0)
            view[name] = self.encode_front(interpolate(values, mask, table)[0], grid_mask, weight)
        return view

    def encode_front(self, values, mask, weight):
        """The encoder's front on the encoding grid, (..., G) to (..., G, d). Every ENCODING_POINTS-th point forms
        a grid of rest-frame pixel spacing, encoded as a rest-frame spectrum would be."""
        leading, length = values.shape[:-1], values.shape[-1] // ENCODING_POINTS

        def split(x):
            return x.reshape(-1, length, ENCODING_POINTS).transpose(1, 2).reshape(-1, length)

        features, _ = self.spectral_encoder.front(split(values), split(mask), split(weight))
        features = features.reshape(*leading, ENCODING_POINTS, length, features.shape[-1]).transpose(-3, -2)
        return features.reshape(*leading, ENCODING_POINTS * length, features.shape[-1])

    def scores(self, view, start, stop):
        """Similarity of each view in the rest frame of the redshifts start:stop with the reference spectra in the
        same view.

        Returns
        -------
        list of (torch.Tensor, torch.Tensor)
            For the full and continuum-removed views, the similarity and its validity, of shape
            ``(..., z, classes, phase bins)``.
        """
        lower, upper, weight, valid = self._table("shift", start, stop)
        rest_frame = self._table("rest_frame", start, stop)
        mask = interpolate(view["mask"], view["mask"], rest_frame)[1] * valid
        error = interpolate_error(view["error"], view["mask"], rest_frame)
        reliability = inverse_variance(error, mask) * self.rest_frame_edge_weight[start:stop]
        leading, pixels = mask.shape[:-1], mask.shape[-1]
        flat_mask = mask.reshape(-1, pixels)
        token_weight = pool_weight(
            reliability.reshape(-1, pixels), flat_mask, pixels // self.spectral_encoder.pool_size
        )
        result = []
        for name in VIEWS:
            features = view[name]
            features = features[..., lower, :] * (1.0 - weight[..., None]) + features[..., upper, :] * weight[..., None]
            features = features * valid[..., None]
            tokens, support = self.spectral_encoder.back(
                features.reshape(-1, pixels, features.shape[-1]), flat_mask[:, None]
            )
            references, reference_support, supported, shape = self.references[name]
            similarity, valid_pair = token_cosine(
                tokens,
                support,
                references,
                reference_support,
                supported,
                self.settings["minimum_rest_fraction"],
                self.settings["minimum_shared_fraction"],
                token_weight,
            )
            result.append((similarity.reshape(*leading, *shape), valid_pair.reshape(*leading, *shape)))
        return result
