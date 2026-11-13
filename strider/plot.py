"""The figure of one result (`strider.Classification.plot`), and of results as the spectra arrive
(`strider.Model.plot_accumulation`). They need matplotlib, and use cmasher's colour maps when cmasher is installed."""

import numpy as np

from .bands import EDGES
from .io import _percent
from .posterior import _cell_edges
from .preprocess import network_inputs

__all__ = ["accumulation", "figure", "pyplot"]

BLUE, GREEN, ORANGE, RED, GREY = "#2468A0", "#286044", "#B5562B", "#8E2430", "#777777"
BANDS = (BLUE, GREEN, ORANGE, RED)
STYLE = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}
LABEL = {"ha": "right", "va": "top", "fontsize": 6.5, "bbox": {"fc": "white", "ec": "none", "pad": 1, "alpha": 0.85}}
TITLE_PAD = 30  # room above the top row for (a)'s rest-wavelength axis


def pyplot():
    """matplotlib.pyplot, or an error that says how to install it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('the figure needs matplotlib: pip install "roman-snpit-strider[plot]"') from None
    return plt


def figure(result, name=None):
    """The figure of a result of `strider.Model.classify`: (a) the spectra and their combination, with the four bands
    shaded and the rest wavelength at the main redshift on top; (b) the combined spectrum against the best-matching
    reference of the two most likely classes, each at its own redshift; (c) each visit's brightness in the four bands
    against SN Ia's light curves fitted to it; (d) every class at every redshift; (e) the class probabilities, and the
    redshift posterior of all classes, of SN Ia and of the other classes."""
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
    with plt.rc_context(STYLE), torch.no_grad():
        drawn = plt.figure(figsize=(7.2, 5.6))
        grid = drawn.add_gridspec(2, 3, height_ratios=[1, 1.1], top=0.86, hspace=0.45, wspace=0.3)
        _visits(drawn.add_subplot(grid[0, 0]), result, model, prepared, inputs)
        _references(drawn.add_subplot(grid[0, 1]), result, model, inputs)
        _bands(drawn.add_subplot(grid[0, 2]), result, model, prepared, output)
        _probability_map(drawn, grid[1, :2], result)
        _probabilities(drawn, grid[1, 2], result)
        if name:
            drawn.suptitle(str(name), x=0.06, ha="left", y=0.995, fontsize=9.5)
        _tidy(drawn)
    return drawn


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


def _combined(model, inputs):
    """The combined spectrum, NaN where no visit measures it and at the tapered ends, and its 98th percentile."""
    coadd, _, covered = (_numpy(value) for value in _coadd(inputs))
    shown = np.where((covered > 0) & (_numpy(model.network.matching_weight) > 0.5), coadd, np.nan)
    return shown, np.nanpercentile(np.abs(shown), 98) if np.isfinite(shown).any() else 1.0


def _visits(ax, result, model, prepared, inputs):
    from matplotlib.ticker import MaxNLocator

    shown, norm = _combined(model, inputs)
    keep = np.isfinite(shown)
    level = np.exp(np.log(prepared.scale.astype(np.float64)).mean())
    wavelength = model.package.wavelength_grid / 1e4
    for colour, low, high in zip(BANDS, EDGES[:-1] / 1e4, EDGES[1:] / 1e4):
        ax.axvspan(low, high, color=colour, alpha=0.1, lw=0, zorder=0)
    for flux, scale, measured in zip(prepared.flux, prepared.scale, prepared.measured):
        ax.plot(
            wavelength, np.where(measured & keep, flux * scale / level, np.nan) / norm, color=GREY, lw=0.3, alpha=0.25
        )
    ax.plot(wavelength, shown / norm, color=BLUE, lw=1.0)
    ax.set_ylim(min(-0.2, np.nanpercentile(shown / norm, 1) - 0.2) if keep.any() else -0.2, 1.5)
    ax.set_xlabel(r"Observed wavelength ($\mu$m)")
    ax.set_ylabel("Relative flux")
    z = result.redshift
    rest = ax.secondary_xaxis("top", functions=(lambda x: x / (1.0 + z), lambda x: x * (1.0 + z)))
    rest.set_xlabel(f"Rest wavelength at z = {z:.2f} ($\\mu$m)", labelpad=3)
    rest.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    count = len(prepared.flux)
    ax.set_title(f"(a) {count} spectr{'a' if count > 1 else 'um'}, combined", loc="left", y=1.0, pad=TITLE_PAD)


def _references(ax, result, model, inputs):
    """Each of the two most likely classes at its own redshift: the combined spectrum without its continuum, and the
    class's reference spectrum that matches it best, scaled to it."""
    network = model.network
    coadd, error, covered = _coadd(inputs)
    view = network.matching_view(coadd, covered, error)
    removed = _numpy(view["removed"])
    used = (_numpy(view["mask"]) > 0) & (_numpy(network.matching_weight) > 0.5)
    observed = model.package.wavelength_grid
    rest_grid = _numpy(model.package.spectral_model["rest_wavelength"]).astype(np.float64)
    spectra = _numpy(network.reference_spectra["continuum_removed"])
    masks = _numpy(network.coadd_profile_mask)
    for offset, colour, label in ((2.0, GREEN, result.top_class), (0.0, ORANGE, result.second_class)):
        redshift = result.class_redshifts[label]
        if not np.isfinite(redshift):
            continue
        index = int(np.abs(model.redshift_grid - redshift).argmin())
        score, valid = (_numpy(value) for value in network.phase_bin_scores(view, index))
        k = model.classes.index(label)
        if not valid[k].any():
            continue
        best = int(np.argmax(np.where(valid[k], score[k], -np.inf)))
        rest = observed / (1.0 + redshift)
        reference = np.interp(rest, rest_grid, spectra[k, best], left=np.nan, right=np.nan)
        inside = np.interp(rest, rest_grid, masks[k, best].astype(np.float64), left=0.0, right=0.0) > 0.5
        ok = used & inside & np.isfinite(reference) & np.isfinite(removed)
        if ok.sum() < 2:
            continue
        scale = (removed[ok] @ reference[ok]) / max(reference[ok] @ reference[ok], 1e-300)
        norm = max(np.nanpercentile(np.abs(removed[ok]), 95), 1e-300)
        ax.plot(rest[ok] / 1e4, removed[ok] / norm + offset, color=GREY, lw=0.6)
        ax.plot(rest[ok] / 1e4, scale * reference[ok] / norm + offset, color=colour, lw=1.3)
        ax.text(
            0.98,
            offset + 1.05,
            f"{label} at z = {redshift:.2f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=colour,
        )
    ax.set_ylim(-1.2, 3.6)
    ax.set_yticks([])
    ax.set_xlabel(r"Rest wavelength ($\mu$m)")
    ax.set_title("(b) Best-matching references", loc="left", y=1.0, pad=TITLE_PAD)


def _bands(ax, result, model, prepared, output):
    """Each visit's flux in the four bands, with its error, against SN Ia's light curves fitted to them at its most
    probable redshift and their 68% spread (`strider.brightness.BrightnessTerm.curve`), each band's steady level
    removed, in observer-frame days from the fitted peak; without a fit, the fluxes from the first visit."""
    ia = model.classes.index("Ia")
    row = result.class_redshift_posterior[ia]
    dates = np.linspace(prepared.mjd.min() - 25.0, prepared.mjd.max() + 35.0, 200)
    fit = None
    if np.isfinite(row).all():
        z = float(result.redshift_grid[np.argmax(row)])
        fit = model.brightness.curve(*prepared.band_rows(), ia, z, output["peak_date"][ia], dates)
    flux, error = prepared.band_flux, prepared.band_error
    if fit is None:
        points, days, top = flux, prepared.mjd - prepared.mjd.min(), 0.0
    else:
        points, days, top = flux - fit.level, prepared.mjd - fit.peak_date, np.nanmax(fit.mean + fit.sd)
        ax.text(0.96, 0.94, f"SN Ia at z {z:.2f}", transform=ax.transAxes, **LABEL)
    seen = np.isfinite(points)
    unit = max(top, np.abs(points[seen]).max(), 1e-300) if seen.any() else 1.0
    bottom = min(-0.3, points[seen].min() / unit - 0.1) if seen.any() else -0.3
    for band, colour in enumerate(BANDS):
        if fit is not None:
            shown, mean, sd = dates - fit.peak_date, fit.mean[:, band] / unit, fit.sd[:, band] / unit
            ax.fill_between(shown, mean - sd, mean + sd, color=colour, alpha=0.18, lw=0)
            ax.plot(shown, mean, color=colour, lw=0.9)
        low, high = EDGES[band : band + 2] / 1e4
        mine = seen[:, band]
        ax.errorbar(
            days[mine],
            points[mine, band] / unit,
            error[mine, band] / unit,
            fmt="o",
            ms=3,
            color=colour,
            elinewidth=0.6,
            label=f"{low:.2f}–{high:.2f} $\\mu$m",
        )
    ax.axhline(0.0, color=GREY, lw=0.5, zorder=-5)
    ax.set_xlim(days.min() - 5.0, days.max() + 5.0)
    ax.set_ylim(bottom, 1.35)
    ax.set_yticks([tick for tick in (-1.0, -0.5, 0.0, 0.5, 1.0) if tick > bottom])
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        fontsize=6,
        frameon=False,
        handlelength=0.8,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    ax.set_xlabel("Observer-frame days from peak" if fit is not None else "Observer-frame days")
    ax.set_ylabel("Relative flux")
    ax.set_title("(c) Brightness in four bands", loc="left", y=1.0, pad=TITLE_PAD)


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
        cmap=_colour_map("cmr.ember", "inferno"),
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
    sub = spec.subgridspec(1, 2, width_ratios=[1, 0.025], wspace=0.03)
    ax = fig.add_subplot(sub[0])
    mesh = _map(ax, result, list(result.class_probabilities), values)
    for n, (redshift, _) in enumerate(result.redshift_peaks[:2]):
        cell = int(np.abs(result.redshift_grid - redshift).argmin())
        ax.plot(
            redshift,
            int(np.argmax(values[:, cell])),
            "o",
            ms=8 if n == 0 else 6,
            mfc="none",
            mec="#F2B134" if n == 0 else "#9FD3E6",
            mew=1.2,
            clip_on=False,
            label="main redshift" if n == 0 else "second redshift",
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
    fig.colorbar(mesh, cax=fig.add_subplot(sub[1])).ax.tick_params(labelsize=6)


def _probabilities(fig, spec, result):
    """The three most likely classes, and the redshift posterior of all classes, with its central 68% shaded, of SN Ia
    and of the other classes."""
    right = spec.subgridspec(2, 1, hspace=0.75, height_ratios=[0.75, 1])
    ax = fig.add_subplot(right[0])
    ranked = result._ranked()[:3][::-1]
    ax.barh(range(len(ranked)), [p for _, p in ranked], color=BLUE, height=0.6)
    ax.set_yticks(range(len(ranked)), [label for label, _ in ranked])
    for y, (_, p) in enumerate(ranked):
        if p > 0.8:
            ax.text(p - 0.03, y, _percent(p), va="center", ha="right", fontsize=7, color="white")
        else:
            ax.text(p + 0.03, y, _percent(p), va="center", fontsize=7)
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
    ax.legend(loc="best", fontsize=6, frameon=False, handlelength=1.2, borderaxespad=0.2)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_xlabel("Redshift")
    ax.set_title("Redshift (68% shaded)", loc="left")


def _colour_map(name, fallback):
    """A cmasher colour map, or ``fallback`` without cmasher."""
    try:
        import cmasher  # noqa: F401  (it registers its maps with matplotlib)
    except ImportError:
        return fallback
    return name
