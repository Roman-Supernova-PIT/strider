"""Reading spectra, and writing results.

Many transients go in a table of spectra: a parquet file with one row per visit. Its columns are ``object_id``,
``mjd``, and ``wavelength``, ``flux`` and ``error``, each a list. Optional columns are ``mask``, True for pixels to
ignore, and ``host_redshift`` and ``host_redshift_error``, the same on every row of an object. One transient can also
be a CSV or ECSV file with one row per pixel (`read_transient`), or SMDC spectrum files, one per visit (`read_smdc`).

Results are written in the format the file's ending gives: ``.parquet``, one row per object (`results_table`);
``.csv``, the same rows without the redshift posterior; ``.json``, each object's full result. The evidence maps go
in an HDF5 file (`write_evidence_maps`), and the redshift posteriors in qp's format (`write_redshift_curves`).
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .posterior import central_interval

__all__ = [
    "Visit",
    "Transient",
    "Classification",
    "Redshift",
    "load_example",
    "read_spectra",
    "read_transient",
    "read_smdc",
    "write_spectra",
    "results_schema",
    "results_table",
    "write_results",
    "write_evidence_maps",
    "write_redshift_curves",
]

EXAMPLE = files("strider") / "data" / "example.parquet"
# a transient's table columns, each with the names it may have
COLUMNS = {
    "mjd": ("mjd", "time"),
    "wavelength": ("wavelength",),
    "flux": ("flux",),
    "error": ("error", "flux_err", "fluxerr"),
    "mask": ("mask",),
    "host_redshift": ("host_redshift",),
    "host_redshift_error": ("host_redshift_error",),
}


@dataclass
class Visit:
    """One visit: a prism spectrum of a transient at one date.

    Parameters
    ----------
    wavelength : numpy.ndarray
        Observer-frame wavelength in Å, on any grid. The visit should cover the model's wavelength range, given on
        its model card; one up to 2% short of it at either end is used, and one further short is left out.
    flux, error : numpy.ndarray
        Flux per unit wavelength (f_λ), in any units, and its 1σ error in the same units.
    mjd : float
        The MJD of the visit, in days.
    mask : numpy.ndarray, optional
        True for pixels to ignore.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    mjd: float
    mask: np.ndarray | None = None


@dataclass
class Transient:
    """One transient: its visits, and its host galaxy's redshift if known.

    Parameters
    ----------
    visits : list of strider.Visit
        Its visits, in any order.
    host_redshift, host_redshift_error : float, optional
        The host galaxy's spectroscopic or photometric redshift and its 1σ error. Give both or neither.
    """

    visits: list[Visit]
    host_redshift: float | None = None
    host_redshift_error: float | None = None

    @classmethod
    def from_table(cls, table) -> "Transient":
        """Make a transient from a table with one row per pixel.

        Parameters
        ----------
        table : pandas.DataFrame, astropy.table.Table or dict of array_like
            The columns ``mjd``, ``wavelength`` in Å, ``flux`` and ``error``. Optional columns are ``mask``, True for
            pixels to ignore, and ``host_redshift`` and ``host_redshift_error``, the same on every row. The column
            ``time`` can stand for ``mjd``, and ``flux_err`` or ``fluxerr`` for ``error``; case does not matter.
            The rows with one MJD are one visit. Astropy columns with units are converted: wavelength to Å, flux and
            error to f_λ.

        Returns
        -------
        strider.Transient
        """
        if not hasattr(table, "keys"):
            raise ValueError(
                "give a table with one row per pixel (a pandas DataFrame, an astropy Table or a dict of "
                "columns), or the wavelength, flux, error and mjd of each visit"
            )
        names = {str(name).strip().lower(): name for name in table.keys()}
        found = {
            key: next((names[alias] for alias in aliases if alias in names), None) for key, aliases in COLUMNS.items()
        }
        missing = [key for key in ("mjd", "wavelength", "flux", "error") if found[key] is None]
        if missing:
            raise ValueError(
                f"the table has no {', '.join(missing)} column; its columns are "
                f"{', '.join(map(str, table.keys()))}. Give one row per pixel, with the columns mjd, "
                "wavelength, flux and error, and optionally mask"
            )
        wavelength, flux, error = _in_units(*(table[found[key]] for key in ("wavelength", "flux", "error")))
        mjd = _floats(table[found["mjd"]])
        mask = None if found["mask"] is None else np.asarray(table[found["mask"]], dtype=bool)
        if not len(mjd):
            raise ValueError("the table has no rows")
        if not np.isfinite(mjd).all():
            raise ValueError("a row has no finite mjd; give every row the MJD of its visit")
        dates, first = np.unique(mjd, return_index=True)
        visits = []
        for date in dates[np.argsort(first)]:
            rows = mjd == date
            if len(np.unique(wavelength[rows])) < rows.sum():
                raise ValueError(f"the rows at MJD {date} repeat a wavelength; give each visit its own MJD")
            visits.append(
                Visit(wavelength[rows], flux[rows], error[rows], float(date), None if mask is None else mask[rows])
            )
        host = []
        for key in ("host_redshift", "host_redshift_error"):
            values = [np.nan] if found[key] is None else np.unique(_floats(table[found[key]]))
            if len(values) > 1:
                raise ValueError(f"the table gives {len(values)} values of {key}; give the same on every row")
            host.append(None if np.isnan(values[0]) else float(values[0]))
        return cls(visits, *host)


def _floats(values):
    """Float64 values, NaN where a masked column has no value."""
    return np.ma.filled(np.ma.asarray(values, dtype=np.float64), np.nan)


def _in_units(wavelength, flux, error):
    """Wavelength in Å, and flux and error as f_λ in erg s⁻¹ cm⁻² Å⁻¹ where astropy units are given."""
    if getattr(wavelength, "unit", None) is not None:
        from astropy import units

        wavelength = units.Quantity(_floats(wavelength), wavelength.unit).to_value(units.AA, units.spectral())
    wavelength = _floats(wavelength)
    if getattr(flux, "unit", None) is None:
        return wavelength, _floats(flux), _floats(error)
    from astropy import units

    flam, density = units.erg / units.s / units.cm**2 / units.AA, units.spectral_density(wavelength * units.AA)
    return wavelength, *(
        units.Quantity(_floats(values), getattr(values, "unit", None) or flux.unit).to_value(flam, density)
        for values in (flux, error)
    )


def _wrapped(label, parts, width=79):
    """A card line of ``parts`` joined by " · ", continued on indented lines where it would pass ``width``."""
    lines = [f"{label:<10}{parts[0]}"]
    for part in parts[1:]:
        if len(lines[-1]) + 3 + len(part) > width:
            lines.append(" " * 10 + part)
        else:
            lines[-1] += f" · {part}"
    return lines


def _percent(probability):
    return "<1%" if probability < 0.005 else f"{100 * probability:.0f}%"


class Redshift(NamedTuple):
    """The redshift of a transient if it is of one class (`Classification.redshift_given`): its main peak, its
    central 68% and 95% intervals and its posterior."""

    redshift: float
    redshift_68: tuple[float, float]
    redshift_95: tuple[float, float]
    redshift_posterior: np.ndarray


@dataclass(repr=False)
class Classification:
    """The result for one transient. :doc:`/results` explains each field.

    ``print(result)`` prints its card (`card`), and ``result.plot()`` draws its figure.

    Parameters
    ----------
    model, fingerprint : str
        The model's name, and a code that identifies it: its network, brightness term, priors and settings.
    class_probabilities : dict of str to float
        The probability of each class.
    equal_prior_probabilities : dict of str to float
        The class probabilities if every class were equally common.
    p_ia : float
        The probability of a type Ia supernova.
    redshift : float
        The redshift at the main peak of ``redshift_posterior``: where the most probability lies within 0.1.
    redshift_probability : float
        The probability that the true redshift is within 0.1 of ``redshift``.
    redshift_peaks : list of tuple of float
        Each distinct peak of ``redshift_posterior`` as (redshift, probability within 0.1 of it), the main one
        first.
    redshift_68, redshift_95 : tuple of float
        The central 68% and 95% intervals of the redshift.
    redshift_grid, redshift_cell_widths : numpy.ndarray
        The centre and width of each redshift cell.
    redshift_posterior : numpy.ndarray
        The probability that the redshift is in each cell. It sums to 1. Divide by the cell widths to get a density.
    class_redshift_posterior : numpy.ndarray
        (classes, cells) the redshift posterior if the transient is of each class, in the order of
        ``class_probabilities``; NaN for a class with no posterior at the redshifts allowed. `redshift_given`
        summarises one.
    class_redshifts : dict of str to float
        The redshift at the main peak of each class's redshift posterior, NaN for a class with no posterior at the
        redshifts allowed.
    log_evidence : numpy.ndarray
        (classes, cells) the evidence map: the log evidence for each class and redshift cell, before any prior. It is
        -inf where the model had no training transients. Adding the log of a prior probability of each cell and
        normalising gives the posterior over the cells the model was trained on.
    source_probability : float or None
        The probability that the visits hold a transient at all. None if the model has no source probability.
    untrained_share : float
        The probability that the transient is of a subtype the model never saw at that redshift, within a class it did
        see there. Always 0 for a model with one subtype per class.
    trained_classes : list of str
        The classes the model had training transients of at the redshift shown.
    flags : dict of str to bool
        ``weak_source``, ``several_redshift_peaks``, ``untrained_subtypes``, ``few_classes_trained`` and
        ``brightness_failed`` (the brightness term could not be computed, so the result uses the spectra alone).
        ``weak_source`` is None if the model has no source probability.
    best_match : tuple of str and float
        The class and redshift that match the data best, before any prior.
    next_best_class : str
        The best other class, at its own best redshift.
    match_gap : float
        How far the best match is ahead of the next-best class, in log units; not a probability.
    match_strength : str
        ``high``, ``medium`` or ``low``, from the match gap at levels fixed with the model.
    warnings : list of str
        Why the best match may be less clear than its strength says.
    visits_used : list of int
        The position in the input of each visit used, counting from 0, in time order.
    days_observed : float
        The observer-frame days from the first visit used to the last.
    signal_to_noise : float
        The median S/N per pixel of the visits used, combined on their own pixels.
    dropped_visits : dict of int to str
        The reason each visit was left out, by its position in the input.
    error_scale : float
        The factor the errors were multiplied by before classifying (``error_scale`` of `strider.Model.classify`,
        measured if it was ``"auto"``); 1 if they were used as given.
    redshift_range : tuple of float or None
        The redshift range the result was restricted to (``redshift_range`` of `strider.Model.classify`), or None.
        The redshifts and intervals then lie inside it.
    """

    model: str
    fingerprint: str
    class_probabilities: dict[str, float]
    equal_prior_probabilities: dict[str, float]
    p_ia: float
    redshift: float
    redshift_probability: float
    redshift_peaks: list[tuple[float, float]]
    redshift_68: tuple[float, float]
    redshift_95: tuple[float, float]
    redshift_grid: np.ndarray
    redshift_cell_widths: np.ndarray
    redshift_posterior: np.ndarray
    class_redshift_posterior: np.ndarray
    class_redshifts: dict[str, float]
    log_evidence: np.ndarray
    source_probability: float | None
    untrained_share: float
    trained_classes: list[str]
    flags: dict[str, bool | None]
    best_match: tuple[str, float]
    next_best_class: str
    match_gap: float
    match_strength: str
    warnings: list[str]
    visits_used: list[int]
    days_observed: float
    signal_to_noise: float
    dropped_visits: dict[int, str] = field(default_factory=dict)
    error_scale: float = 1.0
    redshift_range: tuple[float, float] | None = None

    def __str__(self) -> str:
        return self.card()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_source", None)  # the model and the visits, for the figure only
        return state

    def __repr__(self) -> str:
        top, probability = self._ranked()[0]
        return f"<Classification {top} {_percent(probability)}, z {self.redshift:.2f}>"

    def card(self, name: str | None = None) -> str:
        """Return the result as a short card: the most likely classes, the redshift and its 68% range, the match
        strength and any warnings, the data, and any checks that need attention. ``name`` heads it."""
        low, high = self.redshift_68
        digits = 3 if high - low < 0.05 else 2
        head = f"STRIDER · {name or 'transient'}"
        room = 77 - len(self.model)
        head = head if len(head) <= room else head[: room - 1] + "…"
        ranked = self._ranked()[:3]
        classes = " · ".join(f"{label} {_percent(p)}" for n, (label, p) in enumerate(ranked) if n == 0 or p >= 0.01)
        visits = len(self.visits_used)
        data = f"{visits} visit{'s' if visits > 1 else ''}"
        if visits > 1:
            data += f" over {self.days_observed:.0f} days" if self.days_observed >= 1.0 else " on one night"
        if np.isfinite(self.signal_to_noise):
            data += f" · S/N {max(self.signal_to_noise, 0.0):.1f}"
        if self.error_scale != 1.0:
            data += f" · errors × {self.error_scale:.2f}"
        checks = []
        if self.flags["weak_source"]:
            checks.append(f"weak source ({_percent(self.source_probability)})")
        if self.flags["few_classes_trained"]:
            trained = ", ".join(self.trained_classes)
            checks.append(f"trained here: {trained} only" if trained else "no class trained here")
        if self.flags["untrained_subtypes"]:
            checks.append(f"untrained subtypes {_percent(self.untrained_share)}")
        if self.flags["brightness_failed"]:
            checks.append("brightness failed, spectra alone")
        lines = [
            head.ljust(79 - len(self.model)) + self.model,
            f"{'Class':<10}{classes}",
            f"{'Redshift':<10}{self.redshift:.{digits}f} ({_percent(self.redshift_probability)} within 0.1)"
            f" · 68% range {low:.{digits}f}–{high:.{digits}f}",
            *_wrapped("Match", [f"{self.match_strength} strength", *self.warnings]),
            f"{'Data':<10}{data}",
        ]
        if checks:
            lines += _wrapped("Checks", checks)
        return "\n".join(lines)

    def plot(self, path: str | os.PathLike | None = None, name: str | None = None):
        """Draw the result's figure: the spectra and their combination, the best-matching references of the two
        most likely classes, each visit's brightness in the four bands against SN Ia's light curves fitted to it,
        every class at every redshift, and the class probabilities and the redshift posterior of SN Ia and of the
        other classes. Needs matplotlib (``pip install "roman-snpit-strider[plot]"``).

        Parameters
        ----------
        path : str or path-like, optional
            A file to save the figure to, such as ``result.png`` or ``result.pdf``.
        name : str, optional
            A title for the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plot import figure

        drawn = figure(self, name)
        if path is not None:
            drawn.savefig(path, dpi=200, bbox_inches="tight")
        return drawn

    def _ranked(self):
        return sorted(self.class_probabilities.items(), key=lambda item: item[1], reverse=True)

    @property
    def top_class(self) -> str:
        """The most likely class."""
        return self._ranked()[0][0]

    @property
    def top_probability(self) -> float:
        """The probability of the most likely class."""
        return self._ranked()[0][1]

    @property
    def second_class(self) -> str:
        """The second most likely class."""
        return self._ranked()[1][0]

    @property
    def second_probability(self) -> float:
        """The probability of the second most likely class."""
        return self._ranked()[1][1]

    def redshift_given(self, name: str) -> Redshift:
        """Return the redshift if the transient is of class ``name``: its main peak, its central 68% and 95%
        intervals, and its posterior. They are NaN if the class has no probability at any redshift."""
        classes = list(self.class_probabilities)
        if name not in classes:
            raise ValueError(f"unknown class {name!r}; the classes are {', '.join(classes)}")
        posterior = np.asarray(self.class_redshift_posterior[classes.index(name)], dtype=np.float64)
        if not np.isfinite(posterior).all():
            return Redshift(np.nan, (np.nan, np.nan), (np.nan, np.nan), posterior)
        grid = self.redshift_grid
        return Redshift(
            self.class_redshifts[name],
            central_interval(grid, posterior, 0.68, self.redshift_range),
            central_interval(grid, posterior, 0.95, self.redshift_range),
            posterior,
        )

    def to_dict(self) -> dict:
        """Return the result as a dictionary that `json.dumps` can write, without the evidence map."""
        ia = self.redshift_given("Ia")
        summary = {
            "top_class": self.top_class,
            "top_probability": self.top_probability,
            "second_class": self.second_class,
            "second_probability": self.second_probability,
        }
        found = asdict(self)
        del found["log_evidence"]
        return _json_safe(
            summary
            | found
            | {
                "redshift_peaks": [{"redshift": z, "probability": p} for z, p in self.redshift_peaks],
                "best_match": {"class": self.best_match[0], "redshift": self.best_match[1]},
                "redshift_ia": ia.redshift,
                "redshift_ia_68": ia.redshift_68,
                "redshift_ia_95": ia.redshift_95,
                "redshift_grid": np.asarray(self.redshift_grid, dtype=float).tolist(),
                "redshift_cell_widths": np.asarray(self.redshift_cell_widths, dtype=float).tolist(),
                "redshift_posterior": np.asarray(self.redshift_posterior, dtype=float).tolist(),
                "class_redshift_posterior": {
                    name: row.tolist()
                    for name, row in zip(
                        self.class_probabilities, np.asarray(self.class_redshift_posterior, dtype=float)
                    )
                },
                "dropped_visits": {str(index): reason for index, reason in self.dropped_visits.items()},
            }
        )


def _json_safe(value):
    """``value`` with NaN numbers replaced by None, which JSON can hold."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


REQUIRED = ("object_id", "mjd", "wavelength", "flux", "error")
OPTIONAL = ("mask", "host_redshift", "host_redshift_error")


def load_example(batch: bool = False) -> "Transient | dict[str, Transient]":
    """Return the bundled example: a simulated SN Ia at redshift 0.87 with eleven visits. With ``batch=True``, return
    three simulated transients by name."""
    objects = read_spectra(EXAMPLE)
    return objects if batch else objects["sn_ia"]


def read_spectra(path: str | os.PathLike) -> dict[str, Transient]:
    """Read a table of spectra. Returns each transient by object id, with its visits in table order.

    Raises
    ------
    ValueError
        If a column is missing, a visit has no MJD, or an object has different host redshifts on different rows.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    present = set(source.schema_arrow.names)
    missing = [name for name in REQUIRED if name not in present]
    if missing:
        raise ValueError(
            f"{path} is missing the column(s) {', '.join(missing)}. A table of spectra needs one row per "
            f"visit, with the columns {', '.join(REQUIRED)}, and optionally {', '.join(OPTIONAL)}"
        )
    columns = [*REQUIRED, *(name for name in OPTIONAL if name in present)]
    objects, hosts = {}, {}
    for batch in source.iter_batches(columns=columns):
        keys = [str(key) for key in batch.column("object_id").to_pylist()]
        mjd = batch.column("mjd").to_numpy(zero_copy_only=False)
        lists = {name: _rows(batch.column(name)) for name in ("wavelength", "flux", "error", "mask") if name in columns}
        host = [
            batch.column(name).cast(pa.float64()).to_numpy(zero_copy_only=False)
            if name in columns  # all null: NaN
            else np.full(len(keys), np.nan)
            for name in ("host_redshift", "host_redshift_error")
        ]
        for row, key in enumerate(keys):
            if not np.isfinite(mjd[row]):
                raise ValueError(f"{path}: a visit of object {key} has no finite mjd; give every row its MJD")
            mask = lists["mask"][row] if "mask" in lists else None
            visit = Visit(
                lists["wavelength"][row],
                lists["flux"][row],
                lists["error"][row],
                float(mjd[row]),
                None if mask is None else mask.astype(bool),
            )
            objects.setdefault(key, []).append(visit)
            given = tuple(None if np.isnan(value) else float(value) for value in (host[0][row], host[1][row]))
            if hosts.setdefault(key, given) != given:
                raise ValueError(
                    f"{path}: object {key} has different host redshifts on different rows; give the same "
                    "host_redshift and host_redshift_error on every row of an object"
                )
    return {key: Transient(visits, *hosts[key]) for key, visits in objects.items()}


def _rows(column):
    """Each row of a parquet list column as a numpy array (None where null), without Python lists."""
    if column.null_count == len(column):
        return [None] * len(column)
    if hasattr(column.type, "list_size"):  # a fixed-size list column
        import pyarrow as pa

        column = column.cast(pa.list_(column.type.value_type))
    values = column.values.to_numpy(zero_copy_only=False)
    offsets = column.offsets.to_numpy()
    nulls = column.is_null().to_numpy(zero_copy_only=False)
    return [None if null else values[start:stop] for start, stop, null in zip(offsets[:-1], offsets[1:], nulls)]


def read_transient(path: str | os.PathLike) -> Transient:
    """Read one transient from a CSV or ECSV file with one row per pixel. `Transient.from_table` gives the columns.
    Reading ECSV needs astropy."""
    path = Path(path)
    if path.suffix.lower() == ".ecsv":
        try:
            from astropy.table import Table
        except ImportError:
            raise ImportError("reading ECSV files needs astropy: pip install astropy") from None
        return Transient.from_table(Table.read(path, format="ascii.ecsv"))
    if path.suffix.lower() == ".csv":
        import pyarrow.csv

        table = pyarrow.csv.read_csv(path)
        return Transient.from_table({name: table[name].to_numpy() for name in table.column_names})
    raise ValueError(
        f"{path}: read_transient reads a .csv or .ecsv file; for many transients, use a parquet table "
        "of spectra (read_spectra)"
    )


def read_smdc(
    paths: list[str | os.PathLike],
    mjd: list[float],
    host_redshift: float | None = None,
    host_redshift_error: float | None = None,
) -> Transient:
    """Read one transient from SMDC spectrum files, one file per visit. Reading them needs h5py.

    Each file holds a visit's wavelength in Å, flux and 1σ error as ``combined/data/lamb``, ``flam`` and ``func``.
    The files do not give the date, so give the MJD of each.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("reading SMDC spectrum files needs h5py: pip install h5py") from None
    paths, mjd = list(paths), np.atleast_1d(np.asarray(mjd, dtype=np.float64))
    if len(paths) != len(mjd):
        raise ValueError(f"give one MJD for each file: there are {len(paths)} files and {len(mjd)} MJDs")
    visits = []
    for path, date in zip(paths, mjd):
        with h5py.File(path, "r") as spectrum:
            names = ("combined/data/lamb", "combined/data/flam", "combined/data/func")
            absent = [name for name in names if name not in spectrum]
            if absent:
                raise ValueError(f"{path} has no {', '.join(absent)}; an SMDC spectrum file holds {', '.join(names)}")
            visits.append(Visit(*(np.asarray(spectrum[name], dtype=np.float64) for name in names), float(date)))
    return Transient(visits, host_redshift, host_redshift_error)


def write_spectra(objects: dict[str, Transient], path: str | os.PathLike) -> None:
    """Write transients, by object id, to a table of spectra that `read_spectra` can read."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "object_id": key,
            "mjd": float(visit.mjd),
            "wavelength": np.asarray(visit.wavelength, dtype=np.float64).tolist(),
            "flux": np.asarray(visit.flux, dtype=np.float64).tolist(),
            "error": np.asarray(visit.error, dtype=np.float64).tolist(),
            "mask": None if visit.mask is None else np.asarray(visit.mask, dtype=bool).tolist(),
            "host_redshift": transient.host_redshift,
            "host_redshift_error": transient.host_redshift_error,
        }
        for key, transient in objects.items()
        for visit in transient.visits
    ]
    # from_pylist takes its columns from the first row, so every row carries every key
    pq.write_table(pa.Table.from_pylist(rows), path)


def results_schema(model, summary: bool = False):
    """Return the pyarrow schema of the results table of ``model``, a `strider.Model`. Its metadata holds the
    redshift cells' centres and widths, as JSON under ``redshift_grid`` and ``redshift_cell_widths``. A ``summary``
    table, as CSV holds it, has no redshift posterior, and gives the visits as text."""
    import pyarrow as pa

    numbers = ["p_ia"] + [
        f"redshift{kind}{end}"
        for kind in ("", "_ia")
        for end in ("", "_68_lower", "_68_upper", "_95_lower", "_95_upper")
    ]
    numbers += [
        "redshift_probability",
        "second_peak_redshift",
        "second_peak_probability",
        "top_class_redshift",
        "second_class_redshift",
        "source_probability",
        "untrained_share",
        "signal_to_noise",
        "days_observed",
    ]
    fields = [
        ("object_id", pa.string()),
        ("top_class", pa.string()),
        ("top_probability", pa.float64()),
        ("second_class", pa.string()),
        ("second_probability", pa.float64()),
    ]
    fields += [(name, pa.float64()) for name in numbers]
    fields += [
        (name, pa.bool_())
        for name in (
            "several_redshift_peaks",
            "weak_source",
            "untrained_subtypes",
            "few_classes_trained",
            "brightness_failed",
        )
    ]
    text = pa.string() if summary else pa.list_(pa.string())
    fields += [
        ("best_match_class", pa.string()),
        ("best_match_redshift", pa.float64()),
        ("next_best_class", pa.string()),
        ("match_gap", pa.float64()),
        ("match_strength", pa.string()),
        ("warnings", text),
    ]
    fields += [(f"{kind}_{name}", pa.float64()) for kind in ("probability", "equal_prior") for name in model.classes]
    visits = pa.string() if summary else pa.list_(pa.int64())
    fields += [("visits_used", visits), ("dropped_visits", text), ("failure", pa.string())]
    fields += [(name, pa.string()) for name in ("model", "fingerprint", "package_sha256", "strider_version")]
    if not summary:
        fields.append(("redshift_posterior", pa.list_(pa.float64())))
    fields.append(("error_scale", pa.float64()))
    grid = {"redshift_grid": model.redshift_grid, "redshift_cell_widths": model.widths}
    metadata = {key: json.dumps(np.asarray(values, float).tolist()) for key, values in grid.items()}
    return pa.schema(fields, metadata=metadata)


def results_table(results: dict[str, "Classification | str"], model, summary: bool = False):
    """Return a pyarrow table of results, with one row per object.

    ``results`` maps each object id to its `strider.Classification`, or to the reason it could not be classified.
    :doc:`/results` lists the columns. A ``summary`` table (`results_schema`) is the one a CSV file holds.
    """
    import pyarrow as pa

    from . import __version__

    rows = []
    for key, result in results.items():
        row = {
            "object_id": str(key),
            "model": model.name,
            "fingerprint": model.fingerprint,
            "package_sha256": model.files_sha256,
            "strider_version": __version__,
        }
        if isinstance(result, str):
            rows.append(row | {"failure": result})
            continue
        dropped = [f"{index}: {reason}" for index, reason in result.dropped_visits.items()]
        second_peak = result.redshift_peaks[1] if len(result.redshift_peaks) > 1 else (None, None)
        redshifts = {}
        for kind, found in (("", result), ("_ia", result.redshift_given("Ia"))):
            redshifts[f"redshift{kind}"] = found.redshift
            for mass in ("68", "95"):
                low, high = getattr(found, f"redshift_{mass}")
                redshifts[f"redshift{kind}_{mass}_lower"] = low
                redshifts[f"redshift{kind}_{mass}_upper"] = high
        rows.append(
            row
            | {
                "top_class": result.top_class,
                "top_probability": result.top_probability,
                "second_class": result.second_class,
                "second_probability": result.second_probability,
                "p_ia": result.p_ia,
                **redshifts,
                "redshift_probability": result.redshift_probability,
                "second_peak_redshift": second_peak[0],
                "second_peak_probability": second_peak[1],
                "top_class_redshift": result.class_redshifts[result.top_class],
                "second_class_redshift": result.class_redshifts[result.second_class],
                "source_probability": result.source_probability,
                "untrained_share": result.untrained_share,
                "signal_to_noise": result.signal_to_noise,
                "days_observed": result.days_observed,
                **result.flags,
                "best_match_class": result.best_match[0],
                "best_match_redshift": result.best_match[1],
                "next_best_class": result.next_best_class,
                "match_gap": result.match_gap,
                "match_strength": result.match_strength,
                "warnings": "; ".join(result.warnings) if summary else list(result.warnings),
                "visits_used": " ".join(map(str, result.visits_used)) if summary else list(result.visits_used),
                "dropped_visits": "; ".join(dropped) if summary else dropped,
                "redshift_posterior": np.asarray(result.redshift_posterior, dtype=np.float64).tolist(),
                "error_scale": result.error_scale,
            }
            | {f"probability_{name}": value for name, value in result.class_probabilities.items()}
            | {f"equal_prior_{name}": value for name, value in result.equal_prior_probabilities.items()}
        )
    schema = results_schema(model, summary)
    return pa.Table.from_pylist([{name: row.get(name) for name in schema.names} for row in rows], schema=schema)


class ResultsFile:
    """A file of results, written a batch at a time, in the format its ending gives: ``.parquet``, ``.csv`` or
    ``.json``."""

    FORMATS = (".parquet", ".csv", ".json")

    def __init__(self, path: str | os.PathLike, model):
        import pyarrow.csv

        self.path, self.model, self.format = Path(path), model, Path(path).suffix.lower()
        if self.format not in self.FORMATS:
            raise ValueError(f"cannot write results to {path}; give a file ending in {', '.join(self.FORMATS)}")
        self.objects = {}
        if self.format == ".parquet":  # whole parquet files, one per write, so a killed batch keeps what it wrote
            self.parts, self.count = self.path.with_name(self.path.name + ".parts"), 0
            if self.parts.exists():
                raise ValueError(
                    f"{self.parts} is left from a batch that stopped: read its parquet files "
                    "(pyarrow.parquet.read_table) or remove it first"
                )
            self.parts.mkdir()
        elif self.format == ".csv":
            self.writer = pyarrow.csv.CSVWriter(self.path, results_schema(model, summary=True))

    def write(self, results: dict[str, "Classification | str"]) -> None:
        """Add results, by object id."""
        if self.format == ".json":
            from . import __version__

            for key, result in results.items():
                if str(key) in self.objects:
                    raise ValueError(f"two transients have the id {str(key)!r}; give each a distinct id")
                found = (
                    {"failure": result, "model": self.model.name, "fingerprint": self.model.fingerprint}
                    if (isinstance(result, str))
                    else result.to_dict()
                )
                self.objects[str(key)] = found | {
                    "strider_version": __version__,
                    "package_sha256": self.model.files_sha256,
                }
        elif self.format == ".parquet":
            import pyarrow.parquet as pq

            pq.write_table(results_table(results, self.model), self.parts / f"part-{self.count:06d}.parquet")
            self.count += 1
        else:
            self.writer.write_table(results_table(results, self.model, summary=True))

    def close(self) -> None:
        """Finish the file."""
        if self.format == ".json":
            self.path.write_text(json.dumps(self.objects, indent=1) + "\n")
        elif self.format == ".parquet":
            import pyarrow.parquet as pq

            joined = self.path.with_name(self.path.name + ".tmp")
            with pq.ParquetWriter(joined, results_schema(self.model)) as writer:
                for part in sorted(self.parts.glob("part-*.parquet")):
                    writer.write_table(pq.read_table(part))
            os.replace(joined, self.path)
            shutil.rmtree(self.parts)
        else:
            self.writer.close()


def _classified(results):
    found = {key: result for key, result in results.items() if not isinstance(result, str)}
    if not found:
        raise ValueError("no transient was classified, so there is nothing to write")
    return found


class EvidenceMapsFile:
    """The evidence maps of transients, written to an HDF5 file a batch at a time. Writing it needs h5py.

    It holds ``log_evidence`` (transients, classes, cells), each transient's evidence for every class and redshift
    cell before any prior (`Classification.log_evidence`); ``object_id``; ``classes``, ``redshift_grid`` and
    ``redshift_cell_widths``; ``log_survey_prior`` (classes, cells), the model's prior; and ``error_scale``, the
    factor each transient's errors were multiplied by (`Classification.error_scale`). Adding a log prior to an evidence
    map and normalising gives the posterior under that prior over the cells the model was trained on.
    """

    def __init__(self, path: str | os.PathLike, model):
        try:
            import h5py
        except ImportError:
            raise ImportError("writing evidence maps needs h5py: pip install h5py") from None
        shape = (len(model.classes), len(model.redshift_grid))
        self.file = h5py.File(path, "w")
        self.maps = self.file.create_dataset(
            "log_evidence", (0, *shape), np.float32, maxshape=(None, *shape), chunks=(1, *shape), compression="gzip"
        )
        self.ids = self.file.create_dataset("object_id", (0,), h5py.string_dtype(), maxshape=(None,))
        self.scales = self.file.create_dataset("error_scale", (0,), np.float64, maxshape=(None,))
        self.file["classes"] = np.array(model.classes, dtype=h5py.string_dtype())
        self.file["redshift_grid"], self.file["redshift_cell_widths"] = model.redshift_grid, model.widths
        self.file["log_survey_prior"] = model.prior["survey_mass"]
        self.file.attrs["model"], self.file.attrs["fingerprint"] = model.name, model.fingerprint

    def write(self, results: dict[str, "Classification | str"]) -> None:
        """Add the transients classified, by object id."""
        found = {str(key): result for key, result in results.items() if not isinstance(result, str)}
        start = len(self.ids)
        for dataset in (self.maps, self.ids, self.scales):
            dataset.resize(start + len(found), axis=0)
        for row, (key, result) in enumerate(found.items(), start):
            self.maps[row], self.ids[row] = result.log_evidence.astype(np.float32), key
            self.scales[row] = result.error_scale
        self.file.flush()

    def close(self) -> None:
        """Finish the file."""
        self.file.close()


class RedshiftCurvesFile:
    """The redshift posteriors of transients as a qp ensemble of histograms on the redshift cells, the format LSST
    and DESC redshift tools read, with ``object_id`` and the main-peak ``redshift`` alongside. It is written when it
    is closed; until then it keeps each transient's posterior (one number per cell). Writing it needs qp
    (pip install qp-prob)."""

    FORMATS = (".hdf5", ".h5", ".hd5", ".hf5", ".fits")

    def __init__(self, path: str | os.PathLike, model):
        if Path(path).suffix.lower() not in self.FORMATS:
            raise ValueError(f"cannot write redshift curves to {path}; give a file ending in {', '.join(self.FORMATS)}")
        try:
            import qp  # noqa: F401
        except ImportError:
            raise ImportError("writing redshift curves needs qp: pip install qp-prob") from None
        self.path, self.model, self.ids, self.redshifts, self.densities = path, model, [], [], []

    def write(self, results: dict[str, "Classification | str"]) -> None:
        """Add the transients classified, by object id."""
        for key, result in results.items():
            if not isinstance(result, str):
                self.ids.append(str(key))
                self.redshifts.append(result.redshift)
                self.densities.append(result.redshift_posterior / self.model.widths)

    def close(self) -> None:
        """Write the file, if any transient was classified."""
        import qp

        from .posterior import _cell_edges

        if not self.ids:
            return
        ensemble = qp.Ensemble(
            qp.hist,
            data={"bins": _cell_edges(self.model.redshift_grid), "pdfs": np.stack(self.densities).astype(np.float64)},
        )
        ensemble.set_ancil({"object_id": np.array(self.ids), "redshift": np.array(self.redshifts)})
        ensemble.write_to(str(self.path))


def write_evidence_maps(results: dict[str, "Classification | str"], path: str | os.PathLike, model) -> None:
    """Write the evidence maps of the transients classified to an HDF5 file (`EvidenceMapsFile`)."""
    _write(EvidenceMapsFile(path, model), _classified(results))


def write_redshift_curves(results: dict[str, "Classification | str"], path: str | os.PathLike, model) -> None:
    """Write the redshift posteriors of the transients classified as a qp ensemble (`RedshiftCurvesFile`)."""
    _write(RedshiftCurvesFile(path, model), _classified(results))


def _write(output, results):
    try:
        output.write(results)
    finally:
        output.close()


def write_results(results: dict[str, "Classification | str"], path: str | os.PathLike, model) -> None:
    """Write results, by object id, to a file. Its ending gives the format: ``.parquet``, a table of results
    (`results_table`); ``.csv``, its summary rows; ``.json``, each object's full result."""
    _write(ResultsFile(path, model), results)
