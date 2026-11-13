"""Loading a trained STRIDER model and classifying with it."""

import contextlib
import hashlib
import itertools
import multiprocessing
import numbers
import os
import re
import signal
import sys
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import posterior, registry
from .io import Classification, EvidenceMapsFile, RedshiftCurvesFile, ResultsFile, Transient, Visit
from .package import read_package
from .preprocess import Prepared, measured_error_scale, network_inputs, prepare

__all__ = ["Model", "load_model"]

HOST_REDSHIFT_LIMIT = 5.0


class Model:
    """A trained STRIDER model. Create one with `load_model`.

    Attributes
    ----------
    name : str
        The model's name and version, as ``<name>-<version>``.
    fingerprint : str
        A code that identifies the model: its network, brightness term, priors and settings. ``files_sha256``
        identifies every file of its package.
    classes : list of str
        The classes it tells apart.
    redshift_grid : numpy.ndarray
        The redshift at the centre of each redshift cell.
    weak_source_below : float or None
        The source probability below which a result is flagged ``weak_source``. None if the model has no source
        probability.
    package
        The model package it was loaded from.
    """

    def __init__(self, package, network, device="cpu"):
        from .brightness import BrightnessTerm

        self.package, self.network, self.device = package, network, device
        info = package.info
        self.name, self.fingerprint = f"{info['name']}-{info['version']}", info["fingerprint"]
        self.files_sha256 = package.files_sha256
        self.classes = list(info["classes"])
        self.redshift_grid = package.redshift_grid.astype(np.float64)
        self.widths = posterior.cell_widths(package.redshift_grid)
        self.measure = info["redshift_grid"]["prior_measure"]
        self.prior = posterior.prior_change(package.priors, self.classes, self.redshift_grid, self.widths)
        # the fusion gives probability only to cells with training transients, so only they are scored
        tables = package.brightness
        self.brightness = BrightnessTerm(tables, self.redshift_grid, scored=np.isfinite(self.prior["training"]))
        self.profile = tables["profile_wavelength"], tables["profile_sigma"]
        fusion = package.config["fusion"]
        self.fusion = fusion["spectrum_weight"], fusion["brightness_weight"]
        self.match_levels = package.config["match_strength"]
        self.calibration = package.calibration if package.calibration["status"] == "fitted" else None
        self.weak_source_below = (
            None if self.calibration is None else self.calibration["source_probability"]["weak_source_below"]
        )

    @classmethod
    def from_package(cls, path: str | os.PathLike, device: str = "cpu") -> "Model":
        """Load a model from a model package.

        Parameters
        ----------
        path : str or path-like
            The package file (``.tar.gz``) or its unpacked directory.
        device : str
            The PyTorch device to run on, ``cpu`` or ``cuda``. MPS does not work, as it has no float64.

        Returns
        -------
        strider.Model
        """
        if str(device).startswith("mps"):
            raise ValueError("MPS has no float64, which STRIDER needs; use device='cpu' or 'cuda'")
        from .network import Network

        package = read_package(path)
        try:
            network = Network(package.config, package.spectral_model, wavelength_grid=package.wavelength_grid)
            network.load_weights(package.weights)
            network.to(device).eval()
            network.cache_references()
            return cls(package, network, device)
        except ValueError as error:
            raise ValueError(f"model package {package.path}: {error}") from None

    def classify(
        self,
        data,
        flux: Sequence[np.ndarray] | None = None,
        error: Sequence[np.ndarray] | None = None,
        mjd: Sequence[float] | None = None,
        mask: Sequence[np.ndarray | None] | None = None,
        *,
        host_redshift: float | None = None,
        host_redshift_error: float | None = None,
        classes: Sequence[str] | None = None,
        class_prior: Mapping[str, float] | None = None,
        redshift_range: tuple[float, float] | None = None,
        redshift_prior: np.ndarray | None = None,
        host_redshift_outlier: float | None = None,
        error_scale: float | str = 1.0,
    ) -> Classification:
        """Classify one transient and estimate its redshift.

        Parameters
        ----------
        data : strider.Transient, table, or sequence of numpy.ndarray
            The transient: a `strider.Transient`; a table with one row per pixel (`strider.Transient.from_table`);
            or the wavelength of each visit, in Å, with ``flux``, ``error`` and ``mjd``. If all visits share one
            wavelength grid, it can be that one array.
        flux, error : sequence of numpy.ndarray, optional
            One array per visit: the flux per unit wavelength (f_λ), in any units, and its 1σ error in the same units.
        mjd : sequence of float, optional
            The MJD of each visit, in days.
        mask : sequence of numpy.ndarray or None, optional
            One array per visit, True for pixels to ignore. None uses every pixel.
        host_redshift, host_redshift_error : float, optional
            A redshift prior from the host galaxy: its spectroscopic or photometric redshift and 1σ error. Give both or
            neither. They override any stored with the transient.
        classes : sequence of str, optional
            The only classes the transient can be. The probabilities are shared among them.
        class_prior : mapping of str to float, optional
            How common each class is, before the data, in place of the model's survey prior. Give a value for every
            class allowed; they need not sum to 1.
        redshift_range : tuple of float, optional
            The lowest and highest redshift the transient can have.
        redshift_prior : numpy.ndarray, optional
            A probability density for the redshift at each point of `redshift_grid`, such as a photometric
            redshift's P(z), used as given.
        host_redshift_outlier : float, optional
            The chance that the host redshift is wrong, such as a wrong host galaxy or a failed photometric redshift:
            that share of its weight is spread over every redshift, so strong data can overrule a wrong host. The
            default is 0.05; use about 0.03 for a spectroscopic host redshift.
        error_scale : float or "auto"
            A factor, from 0.001 to 1000, to multiply every visit's errors by, for errors that understate the noise.
            ``"auto"`` measures it from the visits kept, never below 1 (`strider.preprocess.measured_error_scale`).
            The default, 1, uses the errors as given.

        Returns
        -------
        strider.Classification
        """
        error_scale = _checked_error_scale(error_scale)
        if flux is None:
            transient = data if isinstance(data, Transient) else Transient.from_table(data)
            data, flux, error, mjd, mask = _visit_columns(transient)
            if host_redshift is None and host_redshift_error is None:
                host_redshift, host_redshift_error = transient.host_redshift, transient.host_redshift_error
        host = _host(host_redshift, host_redshift_error, self.redshift_grid)
        priors = self._priors(classes, class_prior, redshift_range, redshift_prior, host_redshift_outlier)
        prepared, scale = self._scaled(data, flux, error, mjd, mask, error_scale)
        return self._result(self._evaluate(prepared), prepared, host, priors, scale)

    def classify_first(
        self,
        data,
        flux: Sequence[np.ndarray] | None = None,
        error: Sequence[np.ndarray] | None = None,
        mjd: Sequence[float] | None = None,
        mask: Sequence[np.ndarray | None] | None = None,
        *,
        steps: Sequence[int] | None = None,
        host_redshift: float | None = None,
        host_redshift_error: float | None = None,
        classes: Sequence[str] | None = None,
        class_prior: Mapping[str, float] | None = None,
        redshift_range: tuple[float, float] | None = None,
        redshift_prior: np.ndarray | None = None,
        host_redshift_outlier: float | None = None,
        error_scale: float | str = 1.0,
    ) -> dict[int, Classification]:
        """Classify the first spectra in time order, for each number of spectra in ``steps``: how the result builds up
        as the spectra arrive. Only spectra STRIDER can use count, so a spectrum it leaves out is skipped. Each number
        of spectra is one `classify`.

        Parameters
        ----------
        data, flux, error, mjd, mask
            The transient, as `classify` takes it.
        steps : sequence of int, optional
            The numbers of spectra, in increasing order, such as ``range(3, 9)``. By default 1, 2, 3, 5, 10 and 20
            spectra, those fewer than all of them, then all; with more than 20, 2 is left out to keep six.
        host_redshift, host_redshift_error, classes, class_prior, redshift_range, redshift_prior, host_redshift_outlier
            Your own priors, as `classify` takes them.
        error_scale : float or "auto"
            As `classify` takes it; ``"auto"`` is measured from the spectra each step uses.

        Returns
        -------
        dict of int to strider.Classification
            The result of the first spectra, by their number. Each result's ``visits_used`` gives the positions in
            the input of the spectra it used.
        """
        priors = {
            "host_redshift": host_redshift,
            "host_redshift_error": host_redshift_error,
            "classes": classes,
            "class_prior": class_prior,
            "redshift_range": redshift_range,
            "redshift_prior": redshift_prior,
            "host_redshift_outlier": host_redshift_outlier,
            "error_scale": _checked_error_scale(error_scale),
        }
        transient = _transient(data, flux, error, mjd, mask)
        if host_redshift is None and host_redshift_error is None:
            _host(transient.host_redshift, transient.host_redshift_error, self.redshift_grid)
        else:
            _host(host_redshift, host_redshift_error, self.redshift_grid)
        self._priors(classes, class_prior, redshift_range, redshift_prior, host_redshift_outlier)
        order = self._scaled(*_visit_columns(transient), priors["error_scale"])[0].visits.tolist()
        results = {}
        for k in _steps(steps, len(order)):
            if k == len(order):
                results[k] = self.classify(transient, **priors)
                continue
            results[k] = self.classify(replace(transient, visits=[transient.visits[i] for i in order[:k]]), **priors)
            results[k].visits_used = [order[i] for i in results[k].visits_used]  # their positions in the input
        return results

    def plot_accumulation(
        self,
        data,
        flux: Sequence[np.ndarray] | None = None,
        error: Sequence[np.ndarray] | None = None,
        mjd: Sequence[float] | None = None,
        mask: Sequence[np.ndarray | None] | None = None,
        *,
        steps: Sequence[int] | None = None,
        path: str | os.PathLike | None = None,
        name: str | None = None,
        true_redshift: float | None = None,
        rows: str = "binary",
        host_redshift: float | None = None,
        host_redshift_error: float | None = None,
        classes: Sequence[str] | None = None,
        class_prior: Mapping[str, float] | None = None,
        redshift_range: tuple[float, float] | None = None,
        redshift_prior: np.ndarray | None = None,
        host_redshift_outlier: float | None = None,
        error_scale: float | str = 1.0,
    ):
        """Draw how the result builds up as the spectra arrive (`classify_first`). Each column shows the first
        spectra in time order: their combined spectrum and its S/N, the probability of each class at each redshift,
        and the redshift posterior of SN Ia and of the other classes, with P(Ia) and the main redshift. Needs
        matplotlib.

        Parameters
        ----------
        data, flux, error, mjd, mask
            The transient, as `classify` takes it.
        steps : sequence of int, optional
            How many spectra each column uses, as `classify_first` takes them.
        path : str or path-like, optional
            A file to save the figure to, such as ``arrival.png``.
        name : str, optional
            A title for the figure.
        true_redshift : float, optional
            A redshift to mark, such as a simulated transient's true one.
        rows : str
            ``"binary"`` gives two rows, SN Ia and the other classes summed; ``"all"`` gives a row for each class
            allowed.
        host_redshift, host_redshift_error, classes, class_prior, redshift_range, redshift_prior, host_redshift_outlier
            Your own priors, as `classify` takes them.
        error_scale : float or "auto"
            As `classify_first` takes it.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if rows not in ("binary", "all"):
            raise ValueError(f'rows must be "binary" or "all", not {rows!r}')
        if true_redshift is not None and not (
            isinstance(true_redshift, numbers.Real) and self.redshift_grid[0] <= true_redshift <= self.redshift_grid[-1]
        ):
            raise ValueError(
                f"true_redshift {true_redshift!r} must be a number within the model's redshifts, "
                f"{self.redshift_grid[0]:g} to {self.redshift_grid[-1]:g}"
            )
        from .plot import accumulation, pyplot

        pyplot()  # matplotlib is needed before the classifying starts
        priors = {
            "host_redshift": host_redshift,
            "host_redshift_error": host_redshift_error,
            "classes": classes,
            "class_prior": class_prior,
            "redshift_range": redshift_range,
            "redshift_prior": redshift_prior,
            "host_redshift_outlier": host_redshift_outlier,
            "error_scale": _checked_error_scale(error_scale),
        }
        results = self.classify_first(data, flux, error, mjd, mask, steps=steps, **priors)
        if rows == "all":
            allowed = self._priors(classes, class_prior).get("classes", np.ones(len(self.classes), dtype=bool))
            rows = [name for name, kept in zip(self.classes, allowed) if kept]
        drawn = accumulation(results, rows, true_redshift, name)
        if path is not None:
            drawn.savefig(path, dpi=200, bbox_inches="tight")
        return drawn

    def _prepare(self, data, flux, error, mjd, mask, error_scale=1.0):
        if error_scale != 1.0 and error is not None:
            error = [np.asarray(part, dtype=np.float64) * error_scale for part in error]
        settings, grid = self.package.preprocessing, self.package.wavelength_grid
        return prepare(
            data, flux, error, mjd, mask, settings=settings, grid=grid, error_scale=error_scale, profile=self.profile
        )

    def _scaled(self, data, flux, error, mjd, mask, error_scale):
        """The prepared visits and the factor their errors were multiplied by: ``error_scale``, or, if it is
        ``"auto"``, measured from the visits `prepare` keeps (which visits it keeps does not depend on the factor)."""
        if error_scale != "auto":
            return self._prepare(data, flux, error, mjd, mask, error_scale), error_scale
        prepared = self._prepare(data, flux, error, mjd, mask)
        kept = prepared.visits.tolist()

        def used(values):
            return None if values is None else [values[index] for index in kept]

        shared = len(data) and np.ndim(next(iter(data))) == 0  # one wavelength array for every visit
        scale = measured_error_scale(data if shared else used(data), used(flux), used(error), used(mask))
        if scale == 1.0:
            return prepared, scale
        return self._prepare(data, flux, error, mjd, mask, scale), scale

    def _result(self, output, prepared, host, priors, error_scale):
        result = self._summarise(output, prepared, host, priors)
        result.error_scale = float(error_scale)
        result._source = (self, prepared, output)  # what result.plot() draws
        return result

    def classify_many(
        self,
        objects: Mapping[str, Transient],
        output: str | os.PathLike | None = None,
        chunk: int = 100,
        plots: str | os.PathLike | None = None,
        *,
        evidence_maps: str | os.PathLike | None = None,
        redshift_curves: str | os.PathLike | None = None,
        workers: int = 1,
        batch: int | None = None,
        classes: Sequence[str] | None = None,
        class_prior: Mapping[str, float] | None = None,
        redshift_range: tuple[float, float] | None = None,
        redshift_prior: np.ndarray | None = None,
        host_redshift_outlier: float | None = None,
        error_scale: float | str = 1.0,
        keep: bool = True,
    ) -> dict[str, Classification | str]:
        """Classify many transients, and optionally write the results to a file.

        Parameters
        ----------
        objects : mapping of str to strider.Transient
            The transients by object id, as `strider.io.read_spectra` returns them.
        output : str or path-like, optional
            The file to write the results to, ending in ``.parquet``, ``.csv`` or ``.json``
            (`strider.io.write_results`).
        chunk : int
            How many objects to classify between writes to ``output``.
        plots : str or path-like, optional
            A folder to save each object's figure in, as ``<object id>.png`` (`strider.Classification.plot`).
        evidence_maps, redshift_curves : str or path-like, optional
            Files to write the evidence maps (`strider.io.EvidenceMapsFile`, HDF5) and the redshift posteriors
            (`strider.io.RedshiftCurvesFile`, qp) to, written as the batch goes like ``output``.
        workers : int
            How many transients to classify at once, each in its own process on one CPU core. With several cores
            this is the fastest way through a batch on a CPU. In a script, call it under
            ``if __name__ == "__main__":``.
        batch : int, optional
            How many transients the network scores together. On a GPU this is the fast way through a batch; the
            default there is 8, and each transient in a batch needs about 1 GB of GPU memory. A batch that runs out
            of GPU memory is split in half and tried again. The default on a CPU is 1. The results are the same as
            one at a time, and on a GPU as on a CPU, up to float rounding. The brightness term runs on the CPU, one
            transient at a time.
        classes, class_prior, redshift_range, redshift_prior, host_redshift_outlier : optional
            Your own priors, as `classify` takes them, for every object.
        error_scale : float or "auto"
            As `classify` takes it, for every object; ``"auto"`` is measured for each object from its own spectra.
        keep : bool
            Keep every result in the returned dict. For a large batch written to files, ``keep=False`` keeps only
            the failures, so memory does not grow with the batch.

        Returns
        -------
        dict
            Each object's result, by object id. If an object cannot be classified, its entry is the reason, as a
            string, and the batch carries on.

        A batch that is stopped (Ctrl-C, or a scheduler's SIGTERM) first writes the transients done. One that is
        killed outright keeps, for a ``.parquet`` output, the transients written so far in ``<output>.parts``.
        """
        priors = {
            "classes": classes,
            "class_prior": class_prior,
            "redshift_range": redshift_range,
            "redshift_prior": redshift_prior,
            "host_redshift_outlier": host_redshift_outlier,
        }
        self._priors(**priors)  # a mistake in them stops the batch before it starts
        error_scale = _checked_error_scale(error_scale)
        if not (isinstance(workers, numbers.Integral) and workers >= 1):
            raise ValueError(f"workers must be a whole number of at least 1, not {workers!r}")
        if workers > 1 and not str(self.device).startswith("cpu"):
            raise ValueError("workers runs on CPU cores: load the model with device='cpu', or use workers=1")
        if batch is None:
            batch = 1 if str(self.device).startswith("cpu") else 8
        if not (isinstance(batch, numbers.Integral) and batch >= 1):
            raise ValueError(f"batch must be a whole number of at least 1, not {batch!r}")
        if workers > 1 and batch > 1:
            raise ValueError("use workers on CPU cores or batch on a GPU, not both")
        if plots is not None:
            from .plot import pyplot

            pyplot()  # matplotlib is needed before the batch starts
            Path(plots).mkdir(parents=True, exist_ok=True)
        chosen = [Path(p).resolve() for p in (output, evidence_maps, redshift_curves) if p is not None]
        if len(set(chosen)) < len(chosen):
            raise ValueError("the results, evidence maps and redshift curves need different files")
        results, writers = {}, []
        with contextlib.ExitStack() as files, _stop_on_sigterm():  # every file is finished, however the batch ends
            for kind, path in (
                (ResultsFile, output),
                (EvidenceMapsFile, evidence_maps),
                (RedshiftCurvesFile, redshift_curves),
            ):
                if path is not None:
                    writers.append(files.enter_context(contextlib.closing(kind(path, self))))
            pending = [{} for _ in writers]  # each file's transients not yet written to it
            try:
                for key, result in self._each(objects, priors, error_scale, plots, workers, batch):
                    if keep or isinstance(result, str):
                        results[key] = result
                    for waiting in pending:
                        waiting[key] = result
                    if pending and len(pending[0]) == chunk:
                        _write(writers, pending)
            finally:  # a stopped batch (an error, Ctrl-C) first writes the transients done
                _write(writers, pending)
        return results

    def _each(self, objects, priors, error_scale, plots, workers, batch=1):
        """Each object's id and its result (`_for_batch`), in order: here, ``batch`` at a time, or from ``workers``
        processes."""
        if workers == 1 and batch == 1:
            for key, transient in objects.items():
                yield key, self._for_batch(key, transient, priors, error_scale, plots)
            return
        if workers == 1:
            items = iter(objects.items())
            while group := list(itertools.islice(items, 4 * batch)):  # sorted by visits within each window
                yield from self._together(group, priors, error_scale, plots, batch)
            return
        items = list(objects.items())
        pool = ProcessPoolExecutor(
            min(workers, len(items)) or 1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_start_worker,
            initargs=(Path(self.package.path).resolve(), self.files_sha256, plots is not None),
        )
        try:  # a dead worker stops the batch with an error rather than leaving it waiting
            jobs = ((key, transient, priors, error_scale, plots) for key, transient in items)
            yield from zip((key for key, _ in items), pool.map(_worker_result, jobs))
        except BaseException:  # a stop: end the workers now, so the transients done are written at once
            for process in list((getattr(pool, "_processes", None) or {}).values()):
                process.terminate()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

    def _together(self, group, priors, error_scale, plots, batch):
        """Each object's id and result (`_finished`), in order, for objects the network scores ``batch`` at a time,
        those with similar numbers of visits together, as each batch is padded to its most visits. An object that
        cannot be prepared, scored or summarised gets its reason, and the others carry on."""
        checked, ready, results = self._priors(**priors), [], {}
        for key, transient in group:
            try:
                transient = transient if isinstance(transient, Transient) else Transient.from_table(transient)
                host = _host(transient.host_redshift, transient.host_redshift_error, self.redshift_grid)
                ready.append((key, *self._scaled(*_visit_columns(transient), error_scale), host))
            except Exception as error:
                results[key] = _reason(error)
        ready.sort(key=lambda entry: len(entry[1].days))
        try:
            outputs = self._scored([prepared for _, prepared, _, _ in ready], batch)
        except Exception:
            outputs = None
        if outputs is None:  # outside the handler, so the failed attempt's memory is freed first
            outputs = self._scored_each([prepared for _, prepared, _, _ in ready])
        for (key, prepared, scale, host), output in zip(ready, outputs):
            if isinstance(output, str):
                results[key] = output
                continue
            try:
                results[key] = self._result(output, prepared, host, checked, scale)
            except Exception as error:
                results[key] = _reason(error)
        for key, _ in group:
            yield key, self._finished(key, results[key], plots)

    def _for_batch(self, key, transient, priors, error_scale, plots):
        """One object's result in a batch (`_finished`), or why it could not be classified."""
        try:
            result = self.classify(transient, **priors, error_scale=error_scale)
        except Exception as error:
            result = _reason(error)
        return self._finished(key, result, plots)

    def _finished(self, key, result, plots):
        """A result in a batch, its figure saved first if ``plots`` is a folder, without the visits it holds."""
        if not isinstance(result, str):
            if plots is not None:
                try:
                    _save_figure(result, Path(plots), key)
                except Exception as error:
                    warnings.warn(f"{key}: the figure could not be saved ({type(error).__name__}: {error})")
            del result._source  # it holds the visits: too much to keep for a batch
        return result

    def _priors(
        self, classes=None, class_prior=None, redshift_range=None, redshift_prior=None, host_redshift_outlier=None
    ):
        """The user's priors, checked, as `strider.posterior.summarise` takes them."""
        found = {}
        allowed = np.ones(len(self.classes), dtype=bool)
        if classes is not None:
            classes = [classes] if isinstance(classes, str) else list(classes)
            if not classes or any(name not in self.classes for name in classes):
                raise ValueError(f"classes {classes} must be some of this model's classes: {', '.join(self.classes)}")
            allowed = np.isin(self.classes, classes)
            found["classes"] = allowed
        if class_prior is not None:
            unknown = sorted(set(class_prior) - set(self.classes))
            missing = [name for name, kept in zip(self.classes, allowed) if kept and name not in class_prior]
            if unknown or missing:
                raise ValueError(
                    f"class_prior must give a value for each class allowed, and only for classes of this "
                    f"model; it lacks {missing} and has the unknown {unknown}"
                )
            share = np.array([float(class_prior.get(name, 0.0)) for name in self.classes]) * allowed
            if not (np.isfinite(share).all() and (share >= 0.0).all() and share.sum() > 0.0):
                raise ValueError(f"class_prior {dict(class_prior)} must be finite and not negative, with some above 0")
            found["class_prior"], found["classes"] = share, allowed & (share > 0.0)
        change = np.zeros(len(self.redshift_grid))
        if redshift_range is not None:
            low, high = (float(value) for value in redshift_range)
            if not low < high:
                raise ValueError(f"redshift_range {tuple(redshift_range)} must be (lowest, highest), lowest first")
            change = posterior.redshift_range_log_prior(self.redshift_grid, low, high)
            found["redshift_bounds"] = (low, high)
            if not np.isfinite(change).any():
                raise ValueError(
                    f"redshift_range {tuple(redshift_range)} is outside the model's redshifts, "
                    f"{self.redshift_grid[0]:g} to {self.redshift_grid[-1]:g}"
                )
        if redshift_prior is not None:
            density = np.asarray(redshift_prior, dtype=np.float64)
            if density.shape != self.redshift_grid.shape or not (
                np.isfinite(density).all() and (density >= 0.0).all() and density.sum() > 0.0
            ):
                raise ValueError(
                    f"redshift_prior must give a density at each of the {len(self.redshift_grid)} "
                    "redshifts of model.redshift_grid, finite and not negative, with some above 0"
                )
            change = change + posterior.redshift_density_log_prior(self.widths, self.measure, density)
        if redshift_range is not None or redshift_prior is not None:
            found["redshift_prior"] = change
        if host_redshift_outlier is not None:
            if not (isinstance(host_redshift_outlier, numbers.Real) and 0.0 <= host_redshift_outlier < 1.0):
                raise ValueError(f"host_redshift_outlier {host_redshift_outlier!r} must be a number from 0 up to 1")
            found["host_outlier"] = float(host_redshift_outlier)
        return found

    def _scored(self, prepared, batch):
        """`_evaluate_many` ``batch`` objects at a time. A batch that runs out of GPU memory is halved and tried again,
        down to one object, and the smaller size kept for the rest."""
        import torch

        out_of_memory = (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError))
        outputs = []
        while len(outputs) < len(prepared):
            part = prepared[len(outputs) : len(outputs) + batch]
            try:
                outputs += self._evaluate_many(part)
                continue
            except out_of_memory:
                if len(part) == 1:
                    raise
            # outside the handler, whose traceback holds the failed attempt's tensors, so they are freed first
            torch.cuda.empty_cache()
            batch = len(part) // 2
            warnings.warn(f"out of GPU memory scoring {len(part)} transients together; trying {batch}")
        return outputs

    def _scored_each(self, prepared):
        """`_scored` one object at a time, an object that fails getting the reason, as a string, in its place."""
        outputs = []
        for one in prepared:
            try:
                outputs += self._scored([one], 1)
            except Exception as error:
                outputs.append(_reason(error))
        return outputs

    def _evaluate(self, prepared: Prepared) -> dict[str, np.ndarray]:
        """The network's outputs for prepared visits (`strider.network.Network.forward`), as numpy arrays, and the
        brightness term, ``brightness``, with its best peak dates, ``peak_date``."""
        return self._evaluate_many([prepared])[0]

    def _evaluate_many(self, prepared: list[Prepared]) -> list[dict[str, np.ndarray]]:
        """`_evaluate` for several objects: the network scores them together (`strider.network.Network.forward_many`),
        and the brightness term each on its own."""
        import torch

        with torch.inference_mode(), _exact_float32(self.device):
            output = self.network.forward_many([network_inputs(p, self.device) for p in prepared])
        output = {key: value.cpu().numpy() for key, value in output.items()}
        return [
            {key: value[n] for key, value in output.items()} | self._brightness(one) for n, one in enumerate(prepared)
        ]

    def _brightness(self, prepared):
        """The brightness term (classes, z) of one transient from each visit's band fluxes, and its best peak dates
        (classes, template redshifts); 0 on the cells it scores, no dates, and ``brightness_failed``, if it cannot be
        computed, which leaves the spectrum alone."""
        import torch

        try:
            score = self.brightness.score(*prepared.band_rows(), prepared.seed)
            return {"brightness": score.log_ratio, "peak_date": score.peak_date[..., 0], "brightness_failed": False}
        except torch.linalg.LinAlgError as failure:
            warnings.warn(f"the brightness term could not be computed ({failure}); the result uses the spectrum alone")
            dates = np.full((len(self.classes), len(self.brightness.redshifts)), np.nan)
            zero = np.where(self.brightness.scored, 0.0, -np.inf)
            return {"brightness": zero, "peak_date": dates, "brightness_failed": True}

    def _summarise(self, output, prepared, host=None, priors=None) -> Classification:
        """The result for one object from its scores (`_evaluate`), with an optional host redshift and its error, and
        the user's priors (`_priors`)."""
        spectrum = np.where(output["support"], output["coadd_term"], posterior.UNSUPPORTED)
        spectrum = posterior.log_posterior(spectrum, self.widths, self.measure)
        scores = spectrum, output["brightness"], self.prior["training"], *self.fusion
        matched = posterior.match(*scores, self.redshift_grid, self.match_levels)
        found = posterior.summarise(
            posterior.fuse(*scores),
            float(output["source_logit"]),
            self.redshift_grid,
            self.widths,
            self.measure,
            self.prior,
            self.calibration,
            host,
            **(priors or {}),
        )
        probabilities = dict(zip(self.classes, found["class_probabilities"].tolist()))
        return Classification(
            model=self.name,
            fingerprint=self.fingerprint,
            class_probabilities=probabilities,
            equal_prior_probabilities=dict(zip(self.classes, found["equal_prior_probabilities"].tolist())),
            p_ia=probabilities["Ia"],
            redshift=found["redshift"],
            redshift_probability=found["redshift_probability"],
            redshift_peaks=found["redshift_peaks"],
            redshift_68=found["redshift_68"],
            redshift_95=found["redshift_95"],
            redshift_grid=self.redshift_grid,
            redshift_cell_widths=self.widths.astype(np.float64),
            redshift_posterior=found["redshift_posterior"],
            class_redshift_posterior=found["class_redshift_posterior"],
            class_redshifts=dict(zip(self.classes, found["class_redshifts"].tolist())),
            log_evidence=found["log_evidence"],
            source_probability=found["source_probability"],
            untrained_share=found["untrained_share"],
            trained_classes=[name for name, kept in zip(self.classes, found["trained_here"]) if kept],
            flags=found["flags"] | {"brightness_failed": bool(output["brightness_failed"])},
            best_match=(self.classes[matched["best_class"]], matched["best_redshift"]),
            next_best_class=self.classes[matched["next_best_class"]],
            match_gap=matched["match_gap"],
            match_strength=matched["match_strength"],
            warnings=matched["warnings"],
            visits_used=prepared.visits.tolist(),
            days_observed=float(prepared.days[-1] - prepared.days[0]),
            signal_to_noise=prepared.signal_to_noise,
            dropped_visits=prepared.dropped,
            redshift_range=found["redshift_range"],
        )


_worker_model = None


def _reason(error):
    return f"{type(error).__name__}: {error}"


@contextlib.contextmanager
def _stop_on_sigterm():
    """A scheduler's SIGTERM stops a batch as Ctrl-C does, so the transients done are written first (main thread
    only, where Python allows a handler)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


def _write(writers, pending):
    """Write each file's waiting transients, emptying its list first, so a stop during a write never writes them
    twice."""
    for n, writer in enumerate(writers):
        waiting, pending[n] = pending[n], {}
        if waiting:
            writer.write(waiting)


@contextlib.contextmanager
def _exact_float32(device):
    """No TF32 on a GPU, which cuDNN's convolutions and the matrix products use by default there, so the results
    match a CPU's up to float32 rounding."""
    import torch

    if not str(device).startswith("cuda"):
        yield
        return
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _start_worker(path, files_sha256, plots):
    """Load the model in a worker process of `Model.classify_many`, to run on one core, and check it is the
    caller's."""
    global _worker_model
    import torch

    torch.set_num_threads(1)
    if plots:
        import matplotlib

        matplotlib.use("Agg")
    _worker_model = Model.from_package(path)
    if _worker_model.files_sha256 != files_sha256:
        raise RuntimeError(f"the model package at {path} changed after it was loaded")


def _worker_result(job):
    return _worker_model._for_batch(*job)


def _save_figure(result, folder, key):
    """Save a result's figure as ``<key>.png`` in ``folder``, the key's characters other than letters, digits, '.',
    '-' and '_' replaced by '_'."""
    import matplotlib.pyplot as plt

    key = str(key)
    safe = re.sub(r"[^\w.-]", "_", key)
    if safe != key:  # so 'a/b' and 'a_b' do not share a file
        safe += "-" + hashlib.sha256(key.encode()).hexdigest()[:8]
    plt.close(result.plot(folder / f"{safe}.png", name=key))


def _visit_columns(transient):
    """A transient's visits as five lists, one entry per visit: wavelength, flux, error, mjd and mask."""
    names = ("wavelength", "flux", "error", "mjd", "mask")
    return [[getattr(visit, name) for visit in transient.visits] for name in names]


def _transient(data, flux, error, mjd, mask):
    """The transient of `Model.classify`'s inputs, each visit's arrays as given."""
    if flux is None:
        return data if isinstance(data, Transient) else Transient.from_table(data)
    if error is None:
        raise ValueError("flux errors are required; give the 1σ error of every pixel")
    if len(data) and np.ndim(next(iter(data))) == 0:  # one wavelength array for every visit, as prepare reads it
        data = [np.asarray(data, dtype=np.float64)] * len(flux)
    parts = (data, flux, error, mask if mask is not None else [None] * len(flux), np.atleast_1d(mjd))
    if len({len(part) for part in parts}) > 1:
        raise ValueError(
            "wavelength, flux, error, mask and mjd must give the same number of visits; they give "
            + ", ".join(str(len(part)) for part in parts)
        )
    return Transient([Visit(w, f, e, t, m) for w, f, e, m, t in zip(*parts)])


def _steps(steps, count):
    """The numbers of spectra `Model.classify_first` uses, of the ``count`` it can use in all."""
    if steps is None:
        fewer = [k for k in (1, 2, 3, 5, 10, 20) if k < count]
        return [k for k in fewer if k != 2 or len(fewer) < 6] + [count]  # at most six
    if isinstance(steps, (numbers.Number, str)):
        raise ValueError(f"steps {steps!r} must be a sequence of numbers of spectra, such as [5] or range(3, 9)")
    steps = list(steps)
    whole = all(isinstance(k, numbers.Integral) and not isinstance(k, bool) and k >= 1 for k in steps)
    if not steps or not whole or steps != sorted(set(steps)):
        raise ValueError(f"steps {steps} must be numbers of spectra, whole and at least 1, in increasing order")
    if steps[-1] > count:
        raise ValueError(f"steps go up to {steps[-1]} spectra, but this transient has {count} that can be used")
    return steps


def _checked_error_scale(value):
    """``error_scale`` checked: ``"auto"``, or a number from 0.001 to 1000."""
    if isinstance(value, str) and value == "auto":
        return value
    real = isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
    if not (real and 1e-3 <= float(value) <= 1e3):
        raise ValueError(f'error_scale {value!r} must be "auto" or a number from 0.001 to 1000')
    return float(value)


def _host(redshift, error, grid):
    """A host redshift and its error given together, or None; refused more than `HOST_REDSHIFT_LIMIT` errors outside
    the grid, where it has no cell to multiply."""
    if redshift is None and error is None:
        return None
    if redshift is None or error is None:
        raise ValueError("give host_redshift and host_redshift_error together, or neither")
    if not all(isinstance(value, numbers.Real) for value in (redshift, error)):
        raise ValueError(f"host_redshift {redshift!r} and host_redshift_error {error!r} must be numbers")
    if not (np.isfinite(redshift) and np.isfinite(error) and error > 0.0):
        raise ValueError(f"host_redshift {redshift} must be finite and host_redshift_error {error} positive")
    reach = HOST_REDSHIFT_LIMIT * error
    if not grid[0] - reach <= redshift <= grid[-1] + reach:
        raise ValueError(
            f"host_redshift {redshift} ± {error} lies more than {HOST_REDSHIFT_LIMIT:g} errors outside "
            f"the model's redshift range, {grid[0]:g} to {grid[-1]:g}; check it, or leave it out"
        )
    return float(redshift), float(error)


def load_model(name: str | os.PathLike | None = None, device: str = "cpu", sha256: str | None = None) -> Model:
    """Load a model. A released model is downloaded the first time you use it.

    Parameters
    ----------
    name : str or path-like, optional
        A released model's name (see ``strider models``), ``"test"`` for the bundled test model, or the path to a
        model package. If left out, the default model.
    device : str
        The PyTorch device to run on, ``cpu`` or ``cuda``.
    sha256 : str, optional
        For a package file given by path, its expected sha256: the file is refused if it differs. A released model
        is always checked against its own.

    Returns
    -------
    strider.Model

    Raises
    ------
    FileNotFoundError
        If ``name`` looks like a path (it has a directory or ends in ``.tar.gz``) and there is no package there.
    ValueError
        If ``name`` is not a released model, or STRIDER cannot read the package.
    """
    named = name is None or str(name) in (*registry.MODELS, registry.TEST_MODEL)
    if not named and (Path(name).exists() or len(Path(name).parts) > 1 or str(name).endswith(".tar.gz")):
        if sha256 is not None:
            if not Path(name).is_file():
                raise ValueError("sha256 checks a package file; give the .tar.gz, not its directory")
            found = hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if found != sha256.lower():
                raise ValueError(f"{name} has sha256 {found}, not the {sha256} expected")
        return Model.from_package(Path(name), device)
    return Model.from_package(registry.fetch(name), device)
