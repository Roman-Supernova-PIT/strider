"""The figure of one result (`strider.Classification.plot`), and of results as the spectra arrive
(`strider.Model.plot_accumulation`), drawn with matplotlib and cmasher's colour maps."""

import textwrap
import warnings

import numpy as np

from .bands import EDGES
from .io import _percent, _tier
from .posterior import _cell_edges
from .preprocess import network_inputs

__all__ = ["accumulation", "figure", "pyplot"]

BLUE, GREEN, ORANGE, GREY = "#2468A0", "#286044", "#B5651D", "#777777"  # SN Ia, the true redshift, other classes
BANDS = ("#4A4FA8", "#2A8A5C", "#D08A2A", "#C0392B")  # the four bands, blue to red
STYLE = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}
BOX = {"fc": "white", "ec": "none", "pad": 1, "alpha": 0.85}
LABEL = {"ha": "right", "va": "top", "fontsize": 6.5, "bbox": BOX}
TEXT = {"ha": "right", "va": "top", "fontsize": 8, "bbox": BOX}
TITLE_PAD = 30  # room above the top row for (a)'s observed-wavelength axis
DETECTED = 2.0  # the S/N above a band's steady level from which (c) fills a point


def pyplot():
    """matplotlib.pyplot, imported when a figure is drawn, with cmasher's colour maps."""
    import cmasher  # noqa: F401  (it registers its maps with matplotlib)
    import matplotlib.pyplot as plt

    return plt


def figure(result, name=None):
    """The figure of a result of `strider.Model.classify`, under a summary of the result: (a) the combined spectrum
    over the spread of the spectra, with the four bands shaded; (b) the combined spectrum against the best match's
    model spectrum; (c) each visit's brightness in the four bands against the best match's light curves fitted to it;
    (d) every class at every redshift; (e) the class probabilities, and the redshift posterior of all classes, of SN
    Ia and of the other classes."""
    import torch

    plt = pyplot()
    source = getattr(result, "_source", None)
    if source is None:
        raise ValueError(
            "this result has no figure: draw it from a result of Model.classify, or give classify_many "
            "a folder for plots"
        )
    model, prepared, output = source
    inputs = _tensors(prepared, model.device)
    notes = [*result.warnings, *result._report_parts()[3]]
    notes = textwrap.wrap("Warnings: " + "; ".join(notes), 105) if notes else []
    heading = HEADING + (0.07 + NOTES * len(notes) if notes else 0.0)  # inches above the panels' titles
    height = heading + 5.75
    with plt.rc_context(STYLE), torch.no_grad():
        drawn = plt.figure(figsize=(7.2, height))
        grid = drawn.add_gridspec(
            2, 3, height_ratios=[1, 0.95], top=1 - heading / height, bottom=0.45 / height, hspace=0.42, wspace=0.3
        )
        spectra, match, bands = (drawn.add_subplot(grid[0, k]) for k in range(3))
        _visits(spectra, result, model, prepared, inputs)
        _best_match(match, result, model, inputs)
        _bands(bands, result, model, prepared, output)
        _probability_map(drawn, grid[1, :2], result)
        _probabilities(drawn, grid[1, 2], result)
        _heading(drawn, result, name, notes, spectra.get_position().x0, bands.get_position().x1, height)
        _tidy(drawn)
    return drawn


HEADING, NOTES = 2.05, 0.2  # the heading's height in inches, and each line of warnings under it
STRENGTH = {"high": "#2E7D32", "medium": "#A68A00", "low": "#B3261E"}
TIER = {"green": STRENGTH["high"], "yellow": STRENGTH["medium"], "orange": "#D35400", "red": STRENGTH["low"]}
EDGE = "#C9CED6"  # the heading box's edge and dividers


def _heading(fig, result, name, notes, left, right, height):
    """The result in a box above the panels, as its report gives it: the name and model, then four tiles (class,
    redshift, match strength, data), each a label, a value and a detail, and any warnings and checks under the box."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    def y(inches):  # inches from the top of the figure, as a figure fraction
        return 1 - inches / height

    ranked = result._ranked()
    (top, p), others = ranked[0], [(label, q) for label, q in ranked[1:3] if q >= 0.01]
    low, high = result.redshift_68
    digits = 3 if high - low < 0.05 else 2
    best, _ = result.best_match
    visits = len(result.visits_used)
    count = f"{visits} visit{'s' if visits > 1 else ''}"
    data = result._report_parts()[2].removeprefix(count).removeprefix(", ").strip()
    tiles = [
        ("Class", f"{_name(top)} {_percent(p)}", TIER[_tier(p)], _listed(others) or "others <1%"),
        ("Redshift", f"{result.redshift:.{digits}f}", "black", f"68%: {low:.{digits}f}–{high:.{digits}f}"),
        (
            "Match",
            result.match_strength.capitalize(),
            STRENGTH[result.match_strength],
            f"best {_name(best)}, next {_name(result.next_best_class)}",
        ),
        ("Data", count, "black", data),
    ]
    fig.text(left, y(0.12), name or "Transient", fontsize=12, fontweight="bold", va="top")
    fig.text(right, y(0.15), result.model, fontsize=8, color="0.45", ha="right", va="top")
    box_top, box_bottom = y(0.42), y(1.42)
    fig.add_artist(
        FancyBboxPatch(
            (left, box_bottom),
            right - left,
            box_top - box_bottom,
            boxstyle="round,pad=0,rounding_size=0.008",
            transform=fig.transFigure,
            fc="#F3F4F6",
            ec=EDGE,
            lw=0.7,
        )
    )
    width = (right - left) / len(tiles)
    for n, (label, value, colour, detail) in enumerate(tiles):
        x = left + n * width + 0.018
        if n:
            fig.add_artist(Line2D([x - 0.018] * 2, [y(0.55), y(1.29)], transform=fig.transFigure, color=EDGE, lw=0.6))
        fig.text(x, y(0.52), label, fontsize=8, color="0.4", va="top")
        fig.text(x, y(0.72), value, fontsize=15, fontweight="bold", color=colour, va="top")
        fig.text(x, y(1.07), "\n".join(textwrap.wrap(detail, 26)), fontsize=8, color="0.3", va="top", linespacing=1.2)
    if notes:
        fig.text(left, y(1.52), "\n".join(notes), fontsize=8.5, color=STRENGTH["medium"], va="top", linespacing=1.4)


def _listed(classes):
    return "   ".join(f"{_name(label)} {_percent(q)}" for label, q in classes)


def accumulation(results, rows="binary", true_redshift=None, name=None):
    """The figure of `strider.Model.plot_accumulation`: a column for each result of `strider.Model.classify_first`,
    with the combined spectrum and its S/N; SN Ia and the other classes summed (``rows="binary"``), or the classes
    named in ``rows``, at every redshift, relative to the most probable; and the redshift posterior of SN Ia and of
    the other classes, with P(Ia) and the main redshift."""
    import torch
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import offset_copy

    plt = pyplot()
    columns = len(results)
    rows_shown = 2 if rows == "binary" else len(rows)
    height = 0.9 + 0.9 * (2.65 + max(0.7, 0.16 * rows_shown))
    with plt.rc_context(STYLE), torch.no_grad():
        drawn = plt.figure(figsize=(1.4 + 1.05 * columns, height))
        grid = drawn.add_gridspec(
            4,
            columns + 1,
            width_ratios=[1] * columns + [0.06],
            height_ratios=[0.9, 0.45, max(0.7, 0.16 * rows_shown), 1.3],
            top=1 - 0.45 / height,
            bottom=0.45 / height,
            hspace=0.06,
            wspace=0.12,
        )
        redshifts = _cell_edges(next(iter(results.values())).redshift_grid)[[0, -1]]
        ticks = MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]).tick_values(*redshifts)
        ticks = ticks[(ticks >= redshifts[0]) & (ticks <= redshifts[1])]
        inner = ticks[:-1] if redshifts[1] - ticks[-1] < 0.02 * np.ptp(redshifts) else ticks  # no two labels touching
        handles = {}
        for column, (k, result) in enumerate(results.items()):
            model, prepared, _ = result._source
            spectrum = drawn.add_subplot(grid[0, column])
            shown, norm = _combined(model, _tensors(prepared, model.device))
            wavelength = model.package.wavelength_grid / 1e4
            finite = np.isfinite(shown)
            spectrum.plot(wavelength, shown / norm, color=BLUE, lw=0.6)
            spectrum.set_xlim(*(wavelength[finite] if finite.any() else wavelength)[[0, -1]])
            spectrum.set_ylim(min(-0.2, np.nanpercentile(shown / norm, 1) - 0.2) if finite.any() else -0.2, 1.5)
            spectrum.set_yticks([])
            spectrum.xaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 2.5, 5, 10]))
            spectrum.tick_params(axis="x", labelsize=6, pad=1.5)
            spectrum.set_title(f"{k} spectr{'um' if k == 1 else 'a'}")
            if np.isfinite(result.signal_to_noise):
                text = f"S/N {max(result.signal_to_noise, 0.0):.1f}"
                spectrum.text(0.96, 0.92, text, transform=spectrum.transAxes, **LABEL)

            names, values = list(result.class_probabilities), _joint(result)
            if rows == "binary":
                names, values = ["SN Ia", "Other"], _ia_and_other(values, names)
            else:
                values = values[[names.index(row) for row in rows]]
                names = ["SN Ia" if row == "Ia" else row for row in rows]
            ax = drawn.add_subplot(grid[2, column])
            mesh = _map(ax, result, names, values)
            ax.tick_params(axis="x", labelbottom=False, length=0)
            ax.tick_params(axis="y", labelleft=column == 0)
            if true_redshift is not None:
                above = offset_copy(ax.get_xaxis_transform(), drawn, y=3.0, units="points")
                ax.plot(true_redshift, 1.0, "v", ms=4.5, color=GREEN, transform=above, clip_on=False)

            ax = drawn.add_subplot(grid[3, column], sharex=ax)
            total = result.redshift_posterior / result.redshift_cell_widths
            handles["All classes"] = ax.fill_between(result.redshift_grid, total, color="0.85", lw=0)
            for part, colour, label in zip(_redshift_parts(result), (BLUE, ORANGE), ("SN Ia", "Other")):
                if part.any():
                    (handles[label],) = ax.plot(result.redshift_grid, part, color=colour, lw=0.8)
            if true_redshift is not None:
                ax.axvline(true_redshift, color=GREEN, ls=":", lw=0.8)
            ax.set_xlim(*redshifts)
            ax.set_xticks(ticks if column == columns - 1 else inner)
            ax.set_ylim(0, 1.5 * max(total.max(), 1e-300))
            ax.set_yticks([])
            text = rf"$P(\mathrm{{Ia}}) = {result.p_ia:.2f}$" + "\n" + rf"$z = {result.redshift:.2f}$"
            ax.text(0.96, 0.93, text, transform=ax.transAxes, linespacing=1.3, **LABEL)
            if column == 0:
                spectrum.set_ylabel("Combined")
                ax.set_ylabel("$P(z)$")
                left = spectrum
        order = [label for label in ("All classes", "SN Ia", "Other") if label in handles]
        ax.legend(
            [handles[label] for label in order],
            order,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=6,
            frameon=False,
            handlelength=1.2,
        )
        drawn.colorbar(mesh, cax=drawn.add_subplot(grid[2, -1])).ax.tick_params(labelsize=6)
        middle = 0.5 * (left.get_position().x0 + spectrum.get_position().x1)
        for above, label in ((spectrum, r"Observed wavelength ($\mu$m)"), (ax, "Redshift")):
            below = offset_copy(drawn.transFigure, drawn, y=-15.0, units="points")
            drawn.text(middle, above.get_position().y0, label, transform=below, ha="center", va="top")
        if name:
            drawn.suptitle(str(name), x=0.06, ha="left", y=1.0, fontsize=9.5)
        _tidy(drawn)
    return drawn


def _tidy(fig):
    """At most about five labelled ticks on each linear axis, and no minor ticks."""
    from matplotlib.ticker import AutoLocator, MaxNLocator

    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "linear" and type(axis.get_major_locator()) is AutoLocator:
                axis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.minorticks_off()


def _tensors(prepared, device):
    return list(network_inputs(prepared, device))


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _coadd(inputs):
    from .network.coadd import inverse_variance_coadd

    return inverse_variance_coadd(*inputs[:4])


def _measured(model, inputs):
    """The combined spectrum, its error, and the pixels to show: measured by a visit, away from the tapered ends, and
    with an error at most ten times the median there (pixels of larger error carry almost no weight)."""
    coadd, error, covered = (np.squeeze(_numpy(value)).astype(np.float64) for value in _coadd(inputs))
    keep = (covered > 0) & (np.squeeze(_numpy(model.network.matching_weight)) > 0.5)
    if keep.any():
        keep &= error <= 10.0 * np.median(error[keep])
    return coadd, error, keep


def _combined(model, inputs):
    """The combined spectrum, NaN where it is not shown (`_measured`), and its 98th percentile."""
    coadd, _, keep = _measured(model, inputs)
    shown = np.where(keep, coadd, np.nan)
    return shown, np.nanpercentile(np.abs(shown), 98) if keep.any() else 1.0


def _smooth(values, use, sigma):
    """``values`` smoothed over the pixels ``use`` marks, with a Gaussian of ``sigma`` pixels."""
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(np.where(use, values, 0.0), sigma) / np.maximum(gaussian_filter1d(use * 1.0, sigma), 1e-3)


def _shown(model, inputs):
    """The combined spectrum, lightly smoothed, where it is shown (`_measured`), its error, the pixels shown, and its
    97th percentile there."""
    coadd, error, keep = _measured(model, inputs)
    shown = np.where(keep, _smooth(coadd, keep, 2), np.nan)
    unit = np.nanpercentile(shown, 97) if keep.any() else 1.0
    return shown, error, keep, unit if unit > 0 else 1.0


def _fill_height(ax, line, *whole):
    """Fit the y-axis to the spectrum ``line`` and the ``whole`` curves, from zero or just below the line's lowest 2%
    (deeper noise runs off the panel) and the whole curves' lowest point, to room for the labels above 1."""
    line = np.ravel(line)[np.isfinite(np.ravel(line))]
    lows = [np.percentile(line, 2)] if line.size else []
    lows += [np.nanmin(curve) for curve in whole if np.isfinite(curve).any()]
    ax.set_ylim(min(0.0, *lows) - 0.08, 1.3)


def _visits(ax, result, model, prepared, inputs):
    """(a) the combined spectrum over the 16-84% range of the spectra, each smoothed and scaled to the combined
    spectrum's total flux where both measure it, in rest wavelength at the main redshift, with the four bands shaded
    and the observed wavelength on top."""
    from matplotlib.ticker import MaxNLocator

    shown, _, keep, unit = _shown(model, inputs)
    z = result.redshift
    rest = model.package.wavelength_grid / (1.0 + z) / 1e4
    for colour, low, high in zip(BANDS, EDGES[:-1], EDGES[1:]):
        ax.axvspan(low / (1.0 + z) / 1e4, high / (1.0 + z) / 1e4, color=colour, alpha=0.1, lw=0, zorder=0)
    count, low = len(prepared.flux), np.nan
    if count > 1:
        low, high = _spread(prepared, shown, keep, unit)
        ax.fill_between(rest, low, high, color="0.75", lw=0, alpha=0.6)
        ax.text(0.97, 0.87, f"{count} spectra: 16–84%", transform=ax.transAxes, color="0.45", **TEXT)
    ax.plot(rest, shown / unit, color=BLUE, lw=1.0)
    ax.text(0.97, 0.95, _line_name(result), transform=ax.transAxes, color=BLUE, **TEXT)
    ax.set_xlim(*(rest[keep] if keep.any() else rest)[[0, -1]])
    _fill_height(ax, shown / unit)
    ax.set_yticks([])
    ax.set_xlabel(f"Rest wavelength at z = {z:.2f} ($\\mu$m)")
    observed = ax.secondary_xaxis("top", functions=(lambda x: x * (1.0 + z), lambda x: x / (1.0 + z)))
    observed.set_xlabel(r"Observed ($\mu$m)", labelpad=3)
    observed.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    ax.set_title("(a) Spectra", loc="left", pad=TITLE_PAD)


def _spread(prepared, shown, keep, unit):
    """The 16th and 84th percentiles, at each pixel, of the spectra, each smoothed and scaled to the combined spectrum
    ``shown``'s total flux where both measure it, in units of ``unit``. A spectrum's own pixels of extreme error (more
    than ten times its median) are left out, so one very noisy spectrum cannot set the spread."""
    spread = np.full(np.shape(prepared.flux), np.nan)
    for k, (flux, measured, log_error) in enumerate(zip(prepared.flux, prepared.measured, prepared.log_error)):
        both = measured & keep
        if both.any():
            error = np.exp(np.asarray(log_error, dtype=np.float64))
            both &= error <= 10.0 * np.median(error[both])
        line = _smooth(flux, both, 5)
        if np.sum(line[both]) > 0:
            spread[k, both] = np.nansum(shown[both]) / np.sum(line[both]) * line[both]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixels no spectrum measures
        return np.nanpercentile(spread / unit, [16, 84], axis=0)


def _best_match(ax, result, model, inputs):
    """(b) the combined spectrum of (a) against the best match's model spectrum, at its redshift and in the phase
    window the comparison scores highest there, scaled to it by least squares over the pixels of small error."""
    network = model.network
    label, z = result.best_match
    k, cell = model.classes.index(label), int(np.abs(model.redshift_grid - z).argmin())
    combined, error, covered = _coadd(inputs)
    view = network.matching_view(combined, covered, error)
    score, valid = (_numpy(value) for value in network.phase_bin_scores(view, cell))
    shown, error, keep, _ = _shown(model, inputs)
    rest = model.package.wavelength_grid / (1.0 + z) / 1e4
    ax.set_title("(b) Best match", loc="left", pad=TITLE_PAD)
    ax.set_xlabel(f"Rest wavelength at z = {z:.2f} ($\\mu$m)")
    ax.set_yticks([])
    if not valid[k].any():
        return
    best = int(np.argmax(np.where(valid[k], score[k], -np.inf)))
    rest_grid = _numpy(model.package.spectral_model["rest_wavelength"]).astype(np.float64) / 1e4
    spectrum, inside = _numpy(network.reference_spectra["full"])[k, best], _numpy(network.coadd_profile_mask)[k, best]
    reference = np.interp(rest, rest_grid, spectrum, left=np.nan, right=np.nan)
    inside = np.interp(rest, rest_grid, inside.astype(np.float64), left=0, right=0)
    region = keep & np.isfinite(reference) & (inside > 0.5)
    quiet = region & (error < 2.5 * np.median(error[keep]))
    if quiet.sum() < 2:
        return
    scale = np.sum(shown[quiet] * reference[quiet]) / max(np.sum(reference[quiet] ** 2), 1e-300)
    unit = max(np.percentile(shown[region], 97), 1e-300)
    line, fitted = np.where(region, shown, np.nan) / unit, np.where(region, scale * reference, np.nan) / unit
    ax.plot(rest, line, color=BLUE, lw=1.1)
    ax.plot(rest, fitted, color="k", lw=0.9)
    ax.text(0.97, 0.95, _line_name(result), transform=ax.transAxes, color=BLUE, **TEXT)
    ax.text(0.97, 0.87, f"{_name(label)} model", transform=ax.transAxes, **TEXT)
    ax.set_xlim(rest[region][0], rest[region][-1])
    _fill_height(ax, line, fitted)


def _bands(ax, result, model, prepared, output):
    """(c) each visit's flux in the four bands, with its error, against the best match's light curves fitted to them
    at its redshift and their 68% spread (`strider.brightness.BrightnessTerm.curve`), each band's steady level
    removed, in observer-frame days from the fitted peak; without a fit, the fluxes from the first visit. Points
    below S/N 2 are open circles. Points whose error is more than ten times the median do not set the scale, so they
    cannot flatten the rest; they may run off the panel."""
    label, z = result.best_match
    k = model.classes.index(label)
    dates = np.linspace(prepared.mjd.min() - 25.0, prepared.mjd.max() + 35.0, 200)
    days, band, flux, error = prepared.band_rows()
    error = error * output["light_curve_error_scale"]  # the errors the term used
    fit = model.brightness.curve(days, band, flux, error, k, z, output["peak_date"][k], dates)
    flux, error = prepared.band_flux, prepared.band_error * output["light_curve_error_scale"]
    if fit is None:
        points, days, top = flux, prepared.mjd - prepared.mjd.min(), 0.0
        reason = "Too few visits to fit" if len(np.unique(prepared.mjd)) < 2 else "No light-curve fit"
        ax.text(0.97, 0.95, reason, transform=ax.transAxes, **TEXT)
    else:
        points, days, top = flux - fit.level, prepared.mjd - fit.peak_date, np.nanmax(fit.mean + fit.sd)
        ax.text(0.97, 0.95, f"{_name(label)} at z = {z:.2f}", transform=ax.transAxes, **TEXT)
    seen = np.isfinite(points)
    ordinary = seen & (error <= 10.0 * np.median(error[seen])) if seen.any() else seen
    unit = max(top, np.abs(points[ordinary]).max(), 1e-300) if ordinary.any() else 1.0
    bottom = max(-1.0, min(-0.3, points[ordinary].min() / unit - 0.1)) if ordinary.any() else -0.3
    detected = points >= DETECTED * error  # filled; the others open, so noise around zero stays quiet
    for band, colour in enumerate(BANDS):
        if fit is not None:
            shown, mean, sd = dates - fit.peak_date, fit.mean[:, band] / unit, fit.sd[:, band] / unit
            ax.fill_between(shown, mean - sd, mean + sd, color=colour, alpha=0.18, lw=0)
            ax.plot(shown, mean, color=colour, lw=0.9)
        low, high = EDGES[band : band + 2] / 1e4
        for filled in (True, False):
            mine = seen[:, band] & (detected[:, band] == filled)
            ax.errorbar(
                days[mine],
                points[mine, band] / unit,
                error[mine, band] / unit,
                fmt="o",
                ms=2.5 if filled else 3.2,
                mfc=colour if filled else "white",
                mew=0.7,
                color=colour,
                elinewidth=0.6,
                capsize=0,
                label=f"{low:.2f}–{high:.2f} $\\mu$m" if filled else None,
            )
    if (seen & ~detected).any():
        ax.annotate(
            f"Open circles: S/N < {DETECTED:g}",
            (0.97, 0.95),
            xycoords="axes fraction",
            xytext=(0, -13),
            textcoords="offset points",
            **{**TEXT, "fontsize": 7, "color": GREY},
        )  # under the label of the fit
    ax.axhline(0.0, color=GREY, lw=0.5, zorder=-5)
    ax.set_xlim(days.min() - 5.0, days.max() + 5.0)
    ax.set_ylim(bottom, 1.35)
    ax.set_yticks([tick for tick in (-1.0, -0.5, 0.0, 0.5, 1.0) if tick > bottom])
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        fontsize=7,
        frameon=False,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    ax.set_xlabel("Observer-frame days from peak" if fit is not None else "Observer-frame days")
    ax.set_ylabel("Relative band flux")
    ax.set_title("(c) Brightness", loc="left", pad=TITLE_PAD)


def _line_name(result):
    """What the blue line of (a) and (b) is: the spectra combined, or the one spectrum."""
    return "Combined" if len(result.visits_used) > 1 else "Spectrum"


def _name(label):
    """A class as the figure names it: SN Ia, not Ia."""
    return "SN Ia" if label == "Ia" else label


def _joint(result):
    """(classes, cells) the probability of each class and redshift cell."""
    probability = np.array(list(result.class_probabilities.values()))[:, None]
    return np.nan_to_num(result.class_redshift_posterior) * probability


def _ia_and_other(values, names):
    """The row of ``values`` for SN Ia, and the sum of the other rows."""
    ia = names.index("Ia")
    return np.stack([values[ia], np.delete(values, ia, 0).sum(0)])


def _redshift_parts(result):
    """The redshift density if the transient is SN Ia, and if it is of another class, each times its probability:
    together, the redshift posterior's density."""
    return _ia_and_other(_joint(result), list(result.class_probabilities)) / result.redshift_cell_widths


def _map(ax, result, names, values):
    """(rows, cells) ``values`` against redshift, relative to their maximum, on a log scale from 10⁻⁴."""
    from matplotlib.colors import LogNorm

    mesh = ax.pcolormesh(
        _cell_edges(result.redshift_grid),
        np.arange(len(names) + 1) - 0.5,
        np.clip(values / max(values.max(), 1e-300), 1e-4, 1.0),
        cmap="cmr.ember",
        norm=LogNorm(1e-4, 1.0),
        shading="flat",
        rasterized=True,
    )
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(range(len(names)), names, fontsize=6)
    ax.tick_params(axis="y", length=0)
    return mesh


def _probability_map(fig, spec, result):
    """Every class at every redshift, relative to the most probable cell, with the main and second redshift peaks
    circled."""
    values = _joint(result)
    sub = spec.subgridspec(1, 3, width_ratios=[1, 0.025, 0.07], wspace=0.03)  # the last, room for (e)'s labels
    ax = fig.add_subplot(sub[0])
    mesh = _map(ax, result, list(result.class_probabilities), values)
    edges = _cell_edges(result.redshift_grid)
    shown = np.flatnonzero((values >= 1e-4 * values.max()).any(0)) if values.max() > 0 else []
    if len(shown):
        low, high = edges[shown[0]], edges[shown[-1] + 1]
        pad = max(0.1, 0.1 * (high - low))
        ax.set_xlim(max(edges[0], low - pad), min(edges[-1], high + pad))
    from matplotlib.patheffects import Normal, Stroke

    for n, (redshift, _) in enumerate(result.redshift_peaks[:2]):
        cell = int(np.abs(result.redshift_grid - redshift).argmin())
        main = n == 0  # a bold ring; the second, smaller, thinner and fainter
        ax.plot(
            redshift,
            int(np.argmax(values[:, cell])),
            "o",
            ms=9 if main else 6,
            mfc="none",
            mec="white",
            mew=1.6 if main else 0.8,
            alpha=1.0 if main else 0.7,
            path_effects=[Stroke(linewidth=3.2 if main else 1.8, foreground="black"), Normal()],  # seen on any cell
            clip_on=False,
            label="main redshift" if main else "second redshift",
        )
    if result.redshift_peaks:
        ax.legend(
            loc="lower right",
            fontsize=6,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
            handletextpad=0.3,
            borderpad=0.4,
        )
    ax.set_xlabel("Redshift")
    ax.set_title("(d) Every class at every redshift, relative to the most probable", loc="left")
    fig.colorbar(mesh, cax=fig.add_subplot(sub[1])).ax.tick_params(labelsize=7)


def _probabilities(fig, spec, result):
    """The three most likely classes, SN Ia in blue, and the redshift posterior of all classes, with its central 68%
    shaded, of SN Ia and of the other classes."""
    right = spec.subgridspec(2, 1, hspace=0.75, height_ratios=[0.75, 1])
    ax = fig.add_subplot(right[0])
    ranked = result._ranked()[:3][::-1]
    colours = [BLUE if label == "Ia" else ORANGE for label, _ in ranked]
    ax.barh(range(len(ranked)), [p for _, p in ranked], color=colours, height=0.6)
    ax.set_yticks(range(len(ranked)), [_name(label) for label, _ in ranked])
    for y, (_, p) in enumerate(ranked):
        if p > 0.8:
            ax.text(p - 0.03, y, _percent(p), va="center", ha="right", fontsize=8, color="white")
        else:
            ax.text(p + 0.03, y, _percent(p), va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis="y", length=0)
    ax.set_title("(e) Class", loc="left")

    ax = fig.add_subplot(right[1])
    grid = result.redshift_grid
    density = result.redshift_posterior / result.redshift_cell_widths
    low, high = result.redshift_68
    peaks = [z for z, _ in result.redshift_peaks[:2]]
    pad = max(0.1, 1.5 * (high - low))
    view = (grid >= min(low, *peaks) - pad) & (grid <= max(high, *peaks) + pad)
    top = max(density[view].max(), 1e-300)
    shade = view & (grid >= low) & (grid <= high)
    ax.fill_between(grid[shade], 0, density[shade] / top, color=GREY, alpha=0.3, lw=0)
    ax.plot(grid[view], density[view] / top, color=GREY, lw=0.6, label="All classes")
    for part, colour, label in zip(_redshift_parts(result), (BLUE, ORANGE), ("SN Ia", "Other")):
        if part.any():
            ax.plot(grid[view], part[view] / top, color=colour, lw=1, label=label)
    ax.legend(loc="upper right", fontsize=7, frameon=False, handlelength=1.2, borderaxespad=0.2)
    ax.set_ylim(0, 1.5)  # room above the curves for the legend
    ax.set_yticks([])
    ax.set_xlabel("Redshift")
    ax.set_title("Redshift", loc="left")
