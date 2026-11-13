"""The brightness term: how a transient's light curves in four bands compare with each class's, at every redshift.

Each visit gives a flux in four fixed observed-frame bands (`strider.bands`). For a class, redshift z and peak date
t0, the templates give each row (visit and band) an expected flux A mu, with A >= 0 the transient's unknown amplitude,
and a template error A e, e ~ N(0, S) with S = W W' + kappa^2 R. W holds the class's loadings (SN Ia only), and
R_ij = psi_i psi_j exp(-|t_i - t_j| / ((1 + z) ell)) [rho + (1 - rho) 1(b_i = b_j)] is correlated over rest phase and
band, so that a departure from the template that runs through the light curve counts once. With probability ``wide``
a transient takes the larger scale kappa_wide. Beyond the templates' phase window a light curve fades from its edge
value, with a spread that grows with the gap.

Each band may also carry a constant offset of any size, such as host light. The whitened contrasts G, an orthonormal
basis orthogonal to the offsets (G Sigma G' = I for the noise Sigma), remove it exactly. With G S G' = U diag(lam) U',
y = U' G f and m = U' G mu, the log likelihood ratio of a source of amplitude A to none is

    g(A) = sum_p [y_p^2 - (y_p - A m_p)^2 / (1 + A^2 lam_p) - ln(1 + A^2 lam_p)] / 2.

It is integrated over A under a half-normal prior on A |m|, the amplitude in S/N units, and over t0 under a flat
prior on a window about the row of highest S/N, refined around each peak in t0 until the peak is resolved. The null
copies, Gaussian noise at the rows' errors with no source, go through the same steps, and the term L(c, z) is the
transient's value less their mean, so that the dates and errors alone favour no class.
"""

from typing import NamedTuple

import numpy as np
import torch

__all__ = ["BrightnessCurve", "BrightnessScore", "BrightnessTerm"]

STEP = 1.0  # rest days between template phases
LATE, EARLY = 50.0, 10.0  # rest days over which a template fades beyond its window's late and early edges
PEAK_SPAN = (-90.0, 30.0)  # peak-date window about the anchor, in rest days at the highest template redshift
T0_STEP = 2.0  # rest days between the first peak dates
REFINE = (5, 50)  # about a peak, this many peak dates are replaced by this many finer ones
MAX_LEVELS = 6
PEAKS, PEAK_RANGE = 3, 10.0  # peaks refined per class and redshift: at most 3, within 10 nats of the best
# grid redshifts beside a template redshift where L bends by over SHARP nats, within REFINE_Z nats of the class's
# best, are scored at their own redshift, not interpolated
SHARP, REFINE_Z = 0.3, 30.0
AMPLITUDE_PRIOR = 100.0  # the half-normal amplitude prior's scale, in S/N units
NULL_COPIES = 8
# the transient's amplitude integral: a scan, Newton steps from its peak, Gauss-Legendre about the peak, over the
# rest of the scan and near zero; the null copies share one composite Gauss-Legendre rule in log amplitude
NEWTON_STEPS = 6
SCAN_NODES = 32
LEGENDRE = np.polynomial.legendre.leggauss(16)
HEAD = np.polynomial.legendre.leggauss(8)
PANEL, PANELS = np.polynomial.legendre.leggauss(4), 16
OUTER_RANGE = 40.0
CUT = 4.0
COPY_RULE = np.polynomial.legendre.leggauss(6)
COPY_PANELS, COPY_RANGE = 16, (1e-4, 10.0)
BUDGET = 1e7  # about the numbers held per batch of trials


class BrightnessScore(NamedTuple):
    """One transient's brightness term.

    Parameters
    ----------
    log_ratio : numpy.ndarray
        (classes, redshifts) L: the log likelihood ratio of the light curves to no source, less the null copies'
        mean; -inf on the cells not scored.
    peak_date : numpy.ndarray
        (classes, template redshifts `BrightnessTerm.redshifts`, 3) the three peak dates of most weight in the
        integral over the peak date, on the days' zero point; NaN where not scored.
    peak_log_weight : numpy.ndarray
        (classes, template redshifts, 3) the log of their weight in that integral.
    """

    log_ratio: np.ndarray
    peak_date: np.ndarray
    peak_log_weight: np.ndarray


class BrightnessCurve(NamedTuple):
    """A class's light curves fitted to one transient (`BrightnessTerm.curve`).

    Parameters
    ----------
    peak_date : float
        The peak date they are drawn at, on the days' zero point.
    mean, sd : numpy.ndarray
        (dates, 4) each band's fitted flux at the dates, without its steady level, and the 1σ spread about it; NaN in
        a band without rows.
    level : numpy.ndarray
        (4,) each band's steady level, to subtract from its fluxes; NaN in a band without rows.
    """

    peak_date: float
    mean: np.ndarray
    sd: np.ndarray
    level: np.ndarray


class BrightnessTerm:
    """The brightness term of each class on a redshift grid.

    Parameters
    ----------
    templates : dict of numpy.ndarray
        ``classes``; ``z_nodes``, the template redshifts as ln(1 + z); ``mean`` and ``psi2`` (classes, bands,
        template redshifts, phases) and ``loadings`` (the same, rank) on rest phases 1 day apart across
        ``window``; ``tau``, the early and late growth scales of the spread beyond the window; and
        ``discrepancy`` (classes, 5), each class's template error kappa, ell (rest days), rho, kappa_wide and wide.
    grid : numpy.ndarray
        The redshifts to score.
    scored : numpy.ndarray, optional
        (classes, redshifts) bool, the cells to score, by default all; the others are -inf.
    device : str
        The PyTorch device.
    """

    def __init__(self, templates, grid, scored=None, device="cpu"):
        self.device = torch.device(device)
        self.classes = [str(c) for c in templates["classes"]]
        self.nodes = np.asarray(templates["z_nodes"], np.float64)
        self.node_t, self.redshifts = self._t(self.nodes), np.expm1(self.nodes)
        self.mean, self.psi2, self.loadings = (self._t(templates[k]) for k in ("mean", "psi2", "loadings"))
        self.window = tuple(float(v) for v in templates["window"])
        self.tau = tuple(float(v) for v in templates["tau"])
        self.error = self._t(templates["discrepancy"])
        self.peak_window = tuple(span * float(np.exp(self.nodes[-1])) for span in PEAK_SPAN)

        n_nodes = len(self.nodes)
        grid = np.asarray(grid, np.float64)
        self.scored = np.ones((len(self.classes), len(grid)), bool) if scored is None else np.asarray(scored, bool)
        self.position = np.interp(np.log1p(grid), self.nodes, np.arange(n_nodes))
        self.k0 = np.minimum(np.floor(self.position).astype(int), n_nodes - 2)
        node = np.arange(n_nodes)
        reads = self.scored @ ((node == self.k0[:, None]) | (node == self.k0[:, None] + 1))
        # the nodes the scored cells read, and one either side so the bend at their edges is measured
        self.needed = reads.copy()
        self.needed[:, 1:] |= reads[:, :-1]
        self.needed[:, :-1] |= reads[:, 1:]

        x, w = (self._t(a) for a in COPY_RULE)
        x, w = (x + 1) / 2, w / 2
        edges = torch.linspace(
            np.log(COPY_RANGE[0]),
            np.log(COPY_RANGE[1] * AMPLITUDE_PRIOR),
            COPY_PANELS + 1,
            dtype=torch.float64,
            device=self.device,
        )
        t = torch.cat([lo + (hi - lo) * x for lo, hi in zip(edges[:-1], edges[1:])])
        du = torch.cat([(hi - lo) * w for lo, hi in zip(edges[:-1], edges[1:])]) * torch.exp(t)
        self.copy_u = torch.cat([COPY_RANGE[0] * x, torch.exp(t)])
        self.copy_log_weight = _log_prior(self.copy_u) + torch.cat([COPY_RANGE[0] * w, du]).log()

    def _t(self, a, dtype=torch.float64):
        return torch.as_tensor(a, dtype=dtype, device=self.device)

    def score(self, days, band, flux, error, seed):
        """The brightness term of one transient.

        Parameters
        ----------
        days, flux, error : array_like
            Each row's date in days (any zero point), flux and error, in one flux unit; rows without a finite flux
            and a positive error are left out.
        band : array_like
            Each row's band, 0 to 3 (`strider.bands`).
        seed : int
            The null copies' seed.

        Returns
        -------
        BrightnessScore
            With at most one date in each band there is no change to compare, and L is 0.
        """
        n_classes, n_nodes = len(self.classes), len(self.nodes)
        dates, log_weight = np.full((n_classes, n_nodes, 3), np.nan), np.full((n_classes, n_nodes, 3), -np.inf)
        data = self._data(days, band, flux, error, seed)
        if data is None:
            return BrightnessScore(np.where(self.scored, 0.0, -np.inf), dates, log_weight)
        cls, node = np.nonzero(self.needed)
        total = torch.full((n_classes, n_nodes, 1 + NULL_COPIES), -torch.inf, dtype=torch.float64, device=self.device)
        total[cls, node], top, top_log = self._peak_integral(data, cls, node.astype(np.float64))
        dates[cls, node], log_weight[cls, node] = top.cpu().numpy() + data["anchor"], top_log.cpu().numpy()

        f = self._t(self.position - self.k0)[None, :, None]
        values = total[:, self.k0] + f * (total[:, self.k0 + 1] - total[:, self.k0])
        real = total[..., 0].cpu().numpy()
        bend = np.zeros_like(real)
        with np.errstate(invalid="ignore"):
            bend[:, 1:-1] = np.abs(real[:, :-2] - 2 * real[:, 1:-1] + real[:, 2:])
        sharp = (bend > SHARP) & (real >= real.max(1, keepdims=True) - REFINE_Z)
        exact = (sharp[:, self.k0] | sharp[:, self.k0 + 1]) & self.scored
        if exact.any():
            cls, cell = np.nonzero(exact)
            values[cls, cell] = self._peak_integral(data, cls, self.position[cell])[0]
        log_ratio = (values[..., 0] - values[..., 1:].mean(-1)).cpu().numpy()
        return BrightnessScore(np.where(self.scored, log_ratio, -np.inf), dates, log_weight)

    def curve(self, days, band, flux, error, cls, redshift, peak_dates, dates):
        """The light curves of one class fitted to one transient's rows, for display.

        At the template redshift nearest ``redshift`` and at its peak date, the rows are f = A (mu + W m + e) + level
        + noise, with e the correlated template error. The amplitude A maximises the likelihood with the bands' levels
        removed, as the term removes them; the levels follow by generalised least squares, and the modes m take their
        mean given the rows. The curves are A (mu + W m), smoothed over half a rest day, with the spread of the
        template error and of the modes. If the peak date puts a row outside the templates' phase window, the best
        date that does not, on a half-day grid, is used instead.

        Parameters
        ----------
        days, band, flux, error : array_like
            The rows, as `score` takes them.
        cls : int
            The class.
        redshift : float
            The redshift.
        peak_dates : numpy.ndarray
            (template redshifts,) the class's peak date at each template redshift, such as the term's best
            (`BrightnessScore`).
        dates : array_like
            The dates to draw the curves at, evenly spaced.

        Returns
        -------
        BrightnessCurve or None
            None with at most one date in each band, or no peak date.
        """
        node = int(np.argmin(np.abs(self.nodes - np.log1p(redshift))))
        data = self._data(days, band, flux, error, 0)
        if data is None or not np.isfinite(peak_dates[node]):
            return None
        position = self._t([float(node)])
        z1 = float(self._z1(position)[0])
        t, bands = data["t"].cpu().numpy(), data["band"].cpu().numpy()
        t0 = peak_dates[node] - data["anchor"]
        first, last = t.max() - self.window[1] * z1, t.min() - self.window[0] * z1
        if first <= last and not first <= t0 <= last:
            inside = np.arange(first, last + 0.25, 0.5)
            fit = self._trials(data, np.full(len(inside), cls), np.full(len(inside), float(node)), inside)[:, 0]
            t0 = float(inside[int(torch.argmax(fit))])
        trial = self._t([cls], torch.long), position, self._t([t0])
        mean, load, R = (value[0].cpu().numpy() for value in self._model(data, *trial))
        kappa = float(self.error[cls, 0])
        S = load @ load.T + kappa**2 * R
        G = data["G"].cpu().numpy()
        lam, U = np.linalg.eigh(G @ S @ G.T)
        y, m = U.T @ (G @ data["flux"]), U.T @ (G @ mean)
        norm = max(m @ m, 1e-300)  # amplitudes 0.03 to 30 times the least-squares one, at least an S/N of 1e-12
        amplitude = max(m @ y / norm, 1e-12 / np.sqrt(norm)) * np.logspace(-1.5, 1.5, 121)
        ratio = 1 + np.outer(amplitude**2, lam)
        A = amplitude[np.argmin((np.log(ratio) + (y - np.outer(amplitude, m)) ** 2 / ratio).sum(1))]
        inverse = np.linalg.inv(np.diag(data["error"] ** 2) + A**2 * S)
        present = np.unique(bands)
        X = (bands[:, None] == present).astype(np.float64)
        r = data["flux"] - A * mean
        level = np.linalg.solve(X.T @ inverse @ X, X.T @ inverse @ r)
        gain = A * load.T @ inverse
        modes, spread = gain @ (r - X @ level), np.eye(load.shape[1]) - A * gain @ load
        dates = np.asarray(dates, np.float64)
        width = 0.5 * STEP * z1 / np.median(np.diff(dates))
        curves, levels = np.full((2, len(dates), self.mean.shape[1]), np.nan), np.full(self.mean.shape[1], np.nan)
        levels[present] = level
        for b in present:
            shown = self._t(np.full(len(dates), b), torch.long)
            mu, W, psi2 = (v[0].cpu().numpy() for v in self._template(self._t(dates - data["anchor"]), shown, *trial))
            var = A**2 * (kappa**2 * psi2 + np.einsum("ij,jk,ik->i", W, spread, W))
            curves[:, :, b] = _smoothed(A * (mu + W @ modes), width), _smoothed(var, width)
        return BrightnessCurve(t0 + data["anchor"], curves[0], np.sqrt(np.clip(curves[1], 0, None)), levels)

    def _data(self, days, band, flux, error, seed):
        """The usable rows, their dates from the anchor (the row of highest S/N once each band's mean is removed), the
        whitened contrasts G and the data vectors in them (the transient, then the null copies); None if no band has
        two dates."""
        days, flux, error = (np.asarray(a, np.float64) for a in (days, flux, error))
        band = np.asarray(band)
        if np.any((band < 0) | (band >= self.mean.shape[1])):
            raise ValueError(f"band must be 0 to {self.mean.shape[1] - 1}")
        good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
        days, band, flux, error = days[good], band[good], flux[good], error[good]
        bands = np.unique(band)
        if all(len(np.unique(days[band == b])) == 1 for b in bands):
            return None
        projected, w = flux.copy(), 1 / error**2
        for b in bands:
            mine = band == b
            projected[mine] -= np.sum(w[mine] * flux[mine]) / np.sum(w[mine])
        anchor = days[np.argmax(projected / error)]
        offsets = (band[:, None] == bands[None, :]) / error[:, None]
        G = self._t(np.linalg.svd(offsets.T)[2][len(bands) :] / error[None, :])
        copies = np.random.default_rng(seed).normal(0.0, error, (NULL_COPIES, len(flux)))
        Gf = G @ self._t(np.vstack([flux, copies])).T
        t = days - anchor
        return dict(
            anchor=anchor,
            flux=flux,
            error=error,
            t=self._t(t),
            band=self._t(band, torch.long),
            G=G,
            GtG=G.T @ G,
            Gf=Gf,
            h0=(Gf**2).sum(0),
            dt=self._t(np.abs(t[:, None] - t[None, :])),
            same=self._t(band[:, None] == band[None, :]),
        )

    def _z1(self, position):
        """1 + z at node positions, linear in ln(1 + z) between nodes."""
        k = position.clamp(0, len(self.nodes) - 1)
        k0 = k.floor().clamp(0, len(self.nodes) - 2)
        i = k0.long()
        return torch.exp(self.node_t[i] + (k - k0) * (self.node_t[i + 1] - self.node_t[i]))

    def _template(self, t, band, cls, position, t0):
        """Each trial's template mean (T, rows), loadings (T, rows, rank) and psi^2 (T, rows) at the rows' dates t and
        bands: linear in phase, then in ln(1 + z); beyond the window, faded from its edge with a spread that grows with
        the gap."""
        lo, hi = self.window
        position = position.clamp(0, len(self.nodes) - 1)
        phase = (t[None, :] - t0[:, None]) / self._z1(position)[:, None]
        pos = (phase.clamp(lo, hi) - lo) / STEP
        i0 = pos.floor().long().clamp(max=self.mean.shape[-1] - 2)
        frac = pos - i0
        k0 = position.floor().long().clamp(0, len(self.nodes) - 2)
        fz = (position - k0)[:, None]
        c, b = cls[:, None], band[None, :]

        def at(table):
            f, g = (w if table.dim() == 4 else w[..., None] for w in (frac, fz))
            ends = []
            for k in (k0[:, None], k0[:, None] + 1):
                left = table[c, b, k, i0]
                ends.append(left + (table[c, b, k, i0 + 1] - left) * f)
            return ends[0] + (ends[1] - ends[0]) * g

        mean, psi2, load = at(self.mean), at(self.psi2), at(self.loadings)
        late, early = (phase - hi).clamp(min=0), (lo - phase).clamp(min=0)
        decay = torch.exp(-late / LATE - early / EARLY)
        growth = (late / self.tau[1]) ** 2 + (early / self.tau[0]) ** 2
        mean = mean * decay
        return mean, load * decay[..., None], psi2 * decay**2 + growth * mean**2

    def _model(self, data, cls, position, t0):
        """Each trial's template mean mu (T, rows), loadings W (T, rows, rank) and R (T, rows, rows)."""
        mean, load, psi2 = self._template(data["t"], data["band"], cls, position, t0)
        ell, rho = self.error[cls, 1, None, None], self.error[cls, 2, None, None]
        K = torch.exp(-data["dt"] / (self._z1(position)[:, None, None] * ell)) * (rho + (1 - rho) * data["same"])
        s = psi2.sqrt()
        return mean, load, s[:, :, None] * K * s[:, None, :]

    def _trials(self, data, cls, position, t0):
        """log of the amplitude integral at each trial (class, node position, peak date from the anchor), for the
        transient and each null copy: (trials, 1 + copies)."""
        rows, vectors = len(data["t"]), data["Gf"].shape[1]
        chunk = max(16, int(BUDGET / max(rows * (8 * vectors + 40) * 2.9, 12 * rows**2)))
        cls, position, t0 = self._t(cls, torch.long), self._t(position), self._t(t0)
        return torch.cat(
            [
                self._log_ratio(data, cls[s : s + chunk], position[s : s + chunk], t0[s : s + chunk])
                for s in range(0, len(cls), chunk)
            ]
        )

    def _log_ratio(self, data, cls, position, t0):
        """log of the amplitude integral (T, 1 + copies), mixed over the template error's two scales."""
        mean, load, R = self._model(data, cls, position, t0)
        WW = load @ load.mT
        kappa, kappa_wide, wide = self.error[cls][:, [0, 3, 4]].T
        G = data["G"]
        # the prior's scale |m| is floored at sqrt(1e-6 tr(G (W W' + R) G')), so a template the offsets nearly
        # absorb tends to its template error's own evidence
        floor = 1e-6 * ((WW + R) * data["GtG"]).sum((-1, -2))
        lam, U = _eigen(G, WW + kappa[:, None, None] ** 2 * R)
        lam_wide, U_wide = lam * (kappa_wide / kappa)[:, None] ** 2, U
        loaded = WW.flatten(1).any(1)  # with loadings, the wide scale has its own eigenvectors
        if loaded.any():
            lam_wide, U_wide = lam_wide.clone(), U.clone()
            lam_wide[loaded], U_wide[loaded] = _eigen(G, WW[loaded] + kappa_wide[loaded, None, None] ** 2 * R[loaded])
        narrow = self._amplitude(data, mean, lam, U, floor)
        broad = self._amplitude(data, mean, lam_wide, U_wide, floor)
        return torch.logaddexp(torch.log1p(-wide)[:, None] + narrow, torch.log(wide)[:, None] + broad)

    def _amplitude(self, data, mean, lam, U, floor):
        """log of the amplitude integral (T, 1 + copies) in the eigenbasis U of G S G' (eigenvalues lam)."""
        m = (U.mT @ (data["G"] @ mean[..., None]))[..., 0]
        y = U.mT @ data["Gf"]
        h0 = data["h0"]
        root = torch.maximum((m**2).sum(-1), floor).sqrt()
        ok = root > 0
        root = torch.where(ok, root, torch.ones_like(root))

        def columns(v):
            return torch.cat([m[..., None] ** 2, m[..., None] * y[..., v], y[..., v] ** 2], -1)

        own, copies = columns(slice(0, 1)), columns(slice(1, None))

        def log_integrand(t):  # in t = ln u, u = A |m|
            u = torch.exp(t)
            return _g(own, lam, h0[:1], u / root[:, None])[..., 0] + _log_prior(u) + t

        # g <= h0 / 2, so the integrand is below e^-40 of its value at 0 beyond AMPLITUDE_PRIOR sqrt(h0 + 80)
        top = torch.maximum(
            (m * y[..., 0]).sum(-1).abs() / root, AMPLITUDE_PRIOR * torch.sqrt(h0[0].clamp(min=0) + 80.0)
        )
        null = _g(copies, lam, h0[1:], self.copy_u / root[:, None]) + self.copy_log_weight[:, None]
        value = torch.cat([_object_integral(log_integrand, top)[:, None], torch.logsumexp(null, 1)], 1)
        return torch.where(ok[:, None], value, torch.zeros_like(value))

    def _peak_integral(self, data, cls, position):
        """L integrated over the peak date at each (class, node position) pair, for the transient and each copy
        (pairs, 1 + copies); and the transient's three peak dates of most weight, from the anchor, with their log
        weights (pairs, 3)."""
        span = self.peak_window[1] - self.peak_window[0]
        z1 = np.exp(np.interp(position, np.arange(len(self.nodes)), self.nodes))
        count = np.ceil(span / (z1 * T0_STEP)).astype(int)
        step = span / (z1 * count)
        k = np.arange(count.max())
        inside = k < count[:, None]
        phase = np.where(inside, self.peak_window[0] / z1[:, None] + (k + 0.5) * step[:, None], np.nan)
        log_w = np.where(inside, np.log(z1 * step / span)[:, None], -np.inf)
        pair, j = np.nonzero(inside)
        values = torch.full((*inside.shape, 1 + NULL_COPIES), -torch.inf, dtype=torch.float64, device=self.device)
        values[pair, j] = self._trials(data, cls[pair], position[pair], phase[pair, j] * z1[pair])
        levels = [(values, self._t(log_w), self._t(phase))]
        segment = inside.shape[1]
        for _ in range(MAX_LEVELS):
            finer = self._refine(data, *levels[-1], segment, cls, position)
            if finer is None:
                break
            levels.append(finer)
            segment = REFINE[1]
        total = torch.full((len(cls), 1 + NULL_COPIES), -torch.inf, dtype=torch.float64, device=self.device)
        for v, w, _ in levels:
            total = torch.logaddexp(total, torch.logsumexp(v + w[..., None], dim=1))
        score = torch.cat([v[..., 0] + w for v, w, _ in levels], -1)
        dates = torch.cat([p for *_, p in levels], -1) * self._t(z1)[:, None]
        top = torch.topk(score, 3, dim=-1)
        return total, torch.gather(dates, -1, top.indices), top.values

    def _refine(self, data, values, log_w, phase, segment, cls, position):
        """A finer level of peak dates (values, log weights, phases) about each of up to PEAKS local maxima of the
        transient's L per pair within PEAK_RANGE nats of its best (by the parabola through the node and its
        neighbours) where the spacing is over 1.6 posterior SDs; None without one. REFINE[1] dates replace the
        REFINE[0] around a maximum, which are set to -inf in ``values``. Dates run in segments of ``segment``."""
        cells, nodes = REFINE
        span = self.peak_window[1] - self.peak_window[0]
        z1 = self._z1(self._t(position))
        real = values[..., 0]
        start = torch.arange(real.shape[-1], device=self.device) % segment
        left, right = real.roll(1, -1), real.roll(-1, -1)
        left[..., start == 0] = -torch.inf
        right[..., start == segment - 1] = -torch.inf
        curve = left - 2 * real + right
        h = torch.exp(log_w) * span / z1[:, None]
        sd = (-curve).clamp(min=1e-300).rsqrt() * h
        peak = (real >= left) & (real >= right) & torch.isfinite(left) & torch.isfinite(right) & (curve < 0)
        height = torch.where(peak, real - (right - left) ** 2 / (8 * curve.clamp(max=-1e-300)), real)
        need = peak & (sd < h / 1.6) & (height >= height.amax(-1, keepdim=True) - PEAK_RANGE)
        need, height, valid = need.cpu().numpy(), height.cpu().numpy(), torch.isfinite(log_w).cpu().numpy()
        chosen = []
        for i in np.flatnonzero(need.any(-1)):
            picked = []
            for j in sorted(np.flatnonzero(need[i]), key=lambda j: -height[i, j]):
                if all(j // segment != k // segment or abs(j - k) >= cells for k in picked):
                    picked.append(j)
                if len(picked) == PEAKS:
                    break
            chosen += [(i, slot, j) for slot, j in enumerate(picked)]
        if not chosen:
            return None
        i, slot, j = (np.array(a) for a in zip(*chosen))
        base = j // segment * segment
        last_valid = base + np.array([valid[a, b : b + segment].sum() for a, b in zip(i, base)]) - 1
        first, last = np.maximum(j - cells // 2, base), np.minimum(j + cells // 2, last_valid)
        for o in range(cells):
            values[i, np.minimum(first + o, last)] = -torch.inf
        hc = h[i, j]
        fine_h = self._t(last - first + 1) * hc / nodes
        fine = (phase[i, first] - hc / 2)[:, None] + (
            torch.arange(nodes, dtype=torch.float64, device=self.device) + 0.5
        ) * fine_h[:, None]
        new = self._trials(
            data, np.repeat(cls[i], nodes), np.repeat(position[i], nodes), (fine * z1[i][:, None]).ravel()
        )
        rows, columns = i[:, None], slot[:, None] * nodes + np.arange(nodes)
        shape = (len(position), PEAKS * nodes)
        level = torch.full((*shape, values.shape[-1]), -torch.inf, dtype=torch.float64, device=self.device)
        level[rows, columns] = new.reshape(len(i), nodes, -1)
        level_w = torch.full(shape, -torch.inf, dtype=torch.float64, device=self.device)
        level_w[rows, columns] = torch.log(z1[i] * fine_h / span)[:, None]
        where = torch.full(shape, torch.nan, dtype=torch.float64, device=self.device)
        where[rows, columns] = fine
        return level, level_w, where


def _smoothed(values, sigma):
    """``values`` smoothed by a Gaussian of ``sigma`` samples, out to 4 sigma, the ends extended by their edge values
    (as scipy.ndimage.gaussian_filter1d with mode "nearest")."""
    radius = int(4.0 * sigma + 0.5)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return np.convolve(np.pad(values, radius, mode="edge"), kernel / kernel.sum(), mode="valid")


def _log_prior(u):
    """The half-normal amplitude prior's log density at u, in S/N units."""
    return np.log(2.0) - 0.5 * np.log(2 * np.pi) - np.log(AMPLITUDE_PRIOR) - u**2 / (2 * AMPLITUDE_PRIOR**2)


def _eigen(G, S):
    """The eigenvalues (at least 0) and eigenvectors of G S G'."""
    B = G @ S @ G.T
    lam, U = torch.linalg.eigh(0.5 * (B + B.mT))
    return lam.clamp(min=0), U


def _g(columns, lam, h0, amp):
    """g(A) (T, N, V) at amplitudes amp (T, N), from the columns (T, P, 1 + 2V) m^2, m y_v and y_v^2 in the eigenbasis
    (eigenvalues lam (T, P)) and each vector's h0 = |y_v|^2 (V,)."""
    a2 = amp**2
    ratio = a2[..., None] * lam[:, None, :]
    s = (1 / (1 + ratio)) @ columns
    V = len(h0)
    q, n, ff = s[..., 0], s[..., 1 : 1 + V], s[..., 1 + V :]
    return (
        -0.5 * (ff - h0) + amp[..., None] * n - 0.5 * (a2 * q)[..., None] - 0.5 * torch.log1p(ratio).sum(-1)[..., None]
    )


def _object_integral(log_integrand, top):
    """log of the integral of exp(log_integrand(t)) over t = ln u, (T,). The integrand can be a broad plateau over
    decades of amplitude, so: a scan for its peak up to ln(2 top), Newton steps from there, Gauss-Legendre in the
    quantile of a two-sided logistic about the peak (each side stopped up to CUT scales out, 2 CUT when narrow),
    composite Gauss-Legendre over the rest of the scan where the integrand is within OUTER_RANGE nats of its peak
    there (a second, broad mode), and Gauss-Legendre in u from 0 to the scan's start."""
    like = dict(dtype=top.dtype, device=top.device)
    share = torch.linspace(0.0, 1.0, SCAN_NODES, **like)
    t = np.log(1e-3) + (torch.log(2 * top)[:, None] - np.log(1e-3)) * share[None, :]
    values = log_integrand(t)
    mode = torch.gather(t, 1, values.argmax(1)[:, None])
    spacing = t[:, 1] - t[:, 0]
    h = 1e-3
    for _ in range(NEWTON_STEPS):
        v = log_integrand(mode + h * mode.new_tensor([-1.0, 0.0, 1.0]))
        d1, d2 = (v[:, 2] - v[:, 0]) / (2 * h), (v[:, 2] - 2 * v[:, 1] + v[:, 0]) / h**2
        step = torch.where(d2 < 0, -d1 / d2.clamp(max=-1e-300), torch.sign(d1) * spacing)
        mode = mode + step.clamp(-spacing, spacing)[:, None]
    sd = torch.where(d2 < 0, (-d2).clamp(min=1e-300).rsqrt(), spacing)[:, None]
    narrow = sd <= 0.25
    left = torch.where(narrow, 0.7 * sd, torch.ones_like(sd))
    right = torch.where(narrow, 0.7 * sd, (0.7 * sd).clamp(0.7, 3.0))
    stop = torch.where(narrow, 2 * CUT, CUT)
    z = torch.arange(1, 2 * CUT + 1, **like)[None, :]
    peak = log_integrand(mode)
    stops = []  # each side stops where the integrand falls most slowly against the logistic
    for sign, scale in ((-1.0, left), (1.0, right)):
        ratio = torch.where(z <= stop, log_integrand(mode + sign * scale * z) - peak + z, torch.inf)
        stops.append(z[0, ratio.argmin(1)][:, None])

    p, wl = (torch.as_tensor(a, **like) for a in LEGENDRE)
    p, wl = (p + 1) / 2, wl / 2
    mass = 2 * left / (left + right)
    parts = []
    for scale, side, lower, share in ((left, stops[0], True, mass), (right, stops[1], False, 2 - mass)):
        cut = torch.sigmoid(-side)
        lo, hi = (cut, 0.5) if lower else (0.5, 1 - cut)
        u = lo + (hi - lo) * p
        x = torch.log(u) - torch.log1p(-u)
        log_q = torch.log(share / scale) - x - 2 * torch.nn.functional.softplus(-x)
        parts.append((mode + scale * x, torch.log((hi - lo) * wl * share) - log_q))

    low_end, high_end = mode - stops[0] * left, mode + stops[1] * right
    floor, ceiling = torch.minimum(low_end, t[:, :1]), torch.maximum(high_end, t[:, -1:])
    outside = (t < low_end) | (t > high_end)
    need = (outside & (values > values.amax(1, keepdim=True) - OUTER_RANGE)).any(1, keepdim=True)
    need = need | (stops[0] < stop) | (stops[1] < stop)
    x, wx = (torch.as_tensor(a, **like) for a in PANEL)
    x, wx = (x + 1) / 2, wx / 2
    for a, b in ((floor, low_end), (high_end, ceiling)):
        edges = a + (b - a) * torch.linspace(0, 1, PANELS + 1, **like)[None, :]
        width = edges[:, 1:] - edges[:, :-1]
        nodes = (edges[:, :-1, None] + width[..., None] * x).reshape(len(a), -1)
        log_w = torch.log((width[..., None] * wx).reshape(len(a), -1).clamp(min=1e-300))
        parts.append((nodes, torch.where(need, log_w, -torch.inf)))

    edge = torch.exp(floor)
    x, wx = (torch.as_tensor(a, **like) for a in HEAD)
    head = edge * (x + 1) / 2
    parts.append((torch.log(head), torch.log(edge * wx / 2) - torch.log(head)))
    nodes = torch.cat([a for a, _ in parts], -1)
    return torch.logsumexp(log_integrand(nodes) + torch.cat([b for _, b in parts], -1), 1)
