"""From the spectrum's and the brightness term's scores to class probabilities, the redshift posterior and the flags.

The network scores the spectrum against every class at every candidate redshift. Normalised, the scores are the
spectrum's posterior q(c, z) under the classes and redshifts it was trained on, the training prior π(c, z). `fuse`
adds the brightness term L(c, z): the log posterior under the training prior is a (log q − log π) + log π + w L,
normalised, with the model's weights a and w, and no probability where π is 0. `summarise`
then:

1. replaces the training prior with the survey prior, giving the class probabilities and P(z);
2. does the same with every class equally common: the equal-prior probabilities.

A host redshift, if given, multiplies (1) and (2), and so do a redshift range and a redshift density. A host
redshift is trusted with a small share, `HOST_OUTLIER` by default, left for its being wrong, so that strong data can
still overrule a wrong one; a density is used as given. A class prior
replaces the survey's mix of classes in (2), and a restriction to some classes leaves the others out of both. Survey
subtypes the model never trained on at a redshift are left out of the prior. ``untrained_share`` is the posterior on
the subtypes missing from classes that were trained there (none, for a model with one subtype per class).

`match` reads the fused score before any prior: the best class and redshift, and how clearly they beat the others.
"""

import numpy as np

__all__ = [
    "redshift_grid",
    "cell_widths",
    "log_posterior",
    "fuse",
    "match",
    "prior_change",
    "host_redshift_log_prior",
    "redshift_range_log_prior",
    "redshift_density_log_prior",
    "change_prior",
    "quantile",
    "central_interval",
    "window_probability",
    "redshift_peaks",
    "class_redshifts",
    "source_probability",
    "summarise",
]

EPSILON = 1.0e-7
# the network's score where no coadd reference can score the class and redshift
UNSUPPORTED = -1.0e4
# untrained_subtypes is flagged above this share of the posterior on survey subtypes missing from classes trained there
UNTRAINED_LIMIT = 0.05
# share of a host redshift's prior spread flat over the grid, for a wrong host galaxy or a failed photometric redshift
HOST_OUTLIER = 0.05
# half-width of the window a redshift's probability is quoted in: the outlier cut, |dz| > 0.1
WINDOW = 0.1
# the match score leaves out cells whose spectrum log posterior is below this (no coadd reference scores them)
MATCHED = -1.0e3
# a second redshift of the best match is a peak at least MODE_SEPARATION from it in ln(1 + z), the score dipping by
# MODE_DIP between; the spectrum or brightness term alone is informative if its best class leads by INFORMATIVE or more
MODE_SEPARATION, MODE_DIP, INFORMATIVE = 0.05, 1.0, 1.0
DAMAGED = "the package is damaged. Download it again"


def redshift_grid(settings):
    """The candidate redshifts (float32) from the ``model`` settings of ``model_config.json``, uniform in z or in
    ln(1 + z)."""
    low, high, bins = settings["redshift_min"], settings["redshift_max"], settings["redshift_bins"]
    if settings["redshift_spacing"] == "log1p":
        return np.expm1(np.linspace(np.log1p(low), np.log1p(high), bins)).astype(np.float32)
    return np.linspace(low, high, bins, dtype=np.float32)


def cell_widths(grid):
    """Widths of the redshift cells centred on the grid points, the end cells halved (float32)."""
    return np.diff(_cell_edges(grid)).astype(np.float32)


def _cell_edges(grid):
    grid = np.asarray(grid, dtype=np.float64)
    return np.concatenate([grid[:1], 0.5 * (grid[:-1] + grid[1:]), grid[-1:]])


def _cell_shares(edges, low, high):
    """The share of each redshift cell between ``low`` and ``high``."""
    return np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low), 0.0, None) / np.diff(edges)


def log_posterior(scores, widths, measure):
    """The normalised log posterior (classes, z) under the training prior from the network's scores.

    ``measure`` is the package's redshift prior measure: ``flat_z`` weights each cell by its width, ``flat_log1p``
    weights the cells equally.
    """
    return _normalise(np.asarray(scores, dtype=np.float64) + _log_measure(widths, measure))


def fuse(spectrum, brightness, training, spectrum_weight, brightness_weight):
    """The log posterior (classes, z) under the training prior from the spectrum and the brightness term:
    a (log q − log π) + log π + w L, normalised.

    ``spectrum`` is log q (`log_posterior`), ``brightness`` L and ``training`` log π (`prior_change`), with a and w
    the weights. It is -inf where π is 0.
    """
    trained = np.isfinite(training)
    base = np.where(trained, training, 0.0)
    fused = spectrum_weight * (spectrum - base) + base + brightness_weight * np.asarray(brightness, dtype=np.float64)
    return _normalise(np.where(trained, fused, -np.inf))


def match(spectrum, brightness, training, spectrum_weight, brightness_weight, grid, levels):
    """The best match of a transient before any prior, and how clearly it beats the others.

    The match score is S = a (log q − log π) + w L, with `fuse`'s arguments, on the cells with training transients
    where the spectrum is scored. The best match is the class and redshift of highest S; the match gap is how far it
    is ahead of the best other class, each at its own best redshift. ``levels`` (the package's ``match_strength``)
    turn the gap into high, medium or low, and set how close a second redshift must be to warn of it.

    Returns
    -------
    dict
        ``best_class`` and ``next_best_class`` (indices), ``best_redshift``, ``match_gap``, ``match_strength`` and
        ``warnings``: a second redshift of the best class within ``levels["second_redshift"]`` of it, and the
        spectrum and the brightness term each alone favouring different classes or redshifts.
    """
    x = np.log1p(np.asarray(grid, dtype=np.float64))
    trained = np.isfinite(training)
    spectral = np.where(
        trained & (spectrum > MATCHED), spectrum_weight * (spectrum - np.where(trained, training, 0.0)), -np.inf
    )
    score = spectral + brightness_weight * np.where(np.isfinite(brightness), brightness, 0.0)
    best, k = _best(score)
    following, gap = _next_best(score, best)
    warnings = []
    second = _second_redshift(score[best], k, x)
    if second is not None and score[best, k] - score[best, second] < levels["second_redshift"]:
        warnings.append(f"second redshift near z {grid[second]:.2f}")
    light = np.where(np.isfinite(spectral) & np.isfinite(brightness), brightness, -np.inf)
    (spectrum_class, spectrum_z), (light_class, light_z) = _best(spectral), _best(light)
    informative = min(_next_best(spectral, spectrum_class)[1], _next_best(light, light_class)[1]) >= INFORMATIVE
    if informative and (spectrum_class != light_class or abs(x[spectrum_z] - x[light_z]) >= MODE_SEPARATION):
        warnings.append("spectra and brightness disagree")
    strength = "high" if gap >= levels["high"] else "medium" if gap >= levels["medium"] else "low"
    return {
        "best_class": best,
        "best_redshift": float(grid[k]),
        "next_best_class": following,
        "match_gap": gap,
        "match_strength": strength,
        "warnings": warnings,
    }


def _best(score):
    """The (class, redshift cell) of the highest score."""
    c, k = np.unravel_index(np.argmax(score), score.shape)
    return int(c), int(k)


def _next_best(score, best):
    """The best class other than ``best``, each at its own best redshift, and how far ``best`` is ahead of it."""
    top = score.max(axis=1)
    other = int(np.argmax(np.delete(top, best)))
    other += other >= best
    return other, float(top[best] - top[other])


def _second_redshift(row, k, x):
    """The highest distinct peak of ``row`` other than the one at cell ``k``, or None: a local maximum (at an end of
    the grid, above its one neighbour) at least `MODE_SEPARATION` from it in ln(1 + z), with the score dipping by
    `MODE_DIP` between them."""
    lowest = np.where(np.isfinite(row), row, np.inf)
    padded = np.r_[-np.inf, row, -np.inf]
    found = None
    for j in range(len(row)):
        if (
            not np.isfinite(row[j])
            or row[j] < padded[j]
            or row[j] < padded[j + 2]
            or abs(x[j] - x[k]) < MODE_SEPARATION
        ):
            continue
        dip = lowest[min(j, k) : max(j, k) + 1].min() <= min(row[j], row[k]) - MODE_DIP
        if dip and (found is None or row[j] > row[found]):
            found = j
    return found


def prior_change(priors, classes, grid, widths):
    """How the survey prior of ``priors.json`` changes the training prior, on the grid, each (classes, z).

    A subtype the model never trained on at a redshift adds nothing to the survey prior there.

    Returns
    -------
    dict of numpy.ndarray
        ``survey``: the log of the survey prior over the training prior, counting only the subtypes with training
        mass, 0 where a class has none; ``equal_classes``: the same with every class equally common; ``untrained``:
        the share of the survey prior in subtypes without training mass; ``training`` and ``survey_mass``: the log of
        the training prior and of the survey prior (its subtypes with training mass) in each cell, -inf where 0.

    Raises
    ------
    ValueError
        If the survey prior gives a class no share.
    """
    subtypes = priors["fine_to_output"]
    names = list(subtypes)
    unknown = sorted(set(subtypes.values()) - set(classes))
    if unknown:
        raise ValueError(f"priors.json maps subtypes onto {unknown}, which are not classes of this model; " + DAMAGED)
    training = _subtype_density(priors["training_population"]["fine"], names, grid, widths)
    survey = _subtype_density(priors["survey"]["fine"], names, grid, widths)
    member = np.array([[subtypes[name] == output for name in names] for output in classes], dtype=np.float64)
    p_train, p_survey = member @ training, member @ survey
    trained = member @ np.where(training > 0.0, survey, 0.0)
    mix = np.array([float(priors["survey"]["class_fraction"][name]) for name in classes])
    absent = [name for name, share in zip(classes, mix) if not share > 0.0]
    if absent:
        raise ValueError(
            f"priors.json gives {', '.join(absent)} no share of the survey prior; the equal-prior "
            "probabilities divide by each class's share, so every class needs one"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        survey_change = np.log(np.where(p_train > 0.0, trained / p_train, 1.0))
        untrained = np.clip(np.where(p_survey > 0.0, 1.0 - trained / p_survey, 0.0), 0.0, 1.0)
    equal_classes = survey_change - np.log(mix / mix.sum())[:, None]
    widths = np.asarray(widths, dtype=np.float64)
    with np.errstate(divide="ignore"):
        training, survey_mass = np.log(p_train * widths), np.log(trained * widths)
    return {
        "survey": survey_change,
        "equal_classes": equal_classes,
        "untrained": untrained,
        "training": training,
        "survey_mass": survey_mass,
    }


def _subtype_density(part, names, grid, widths):
    """p(s) n(z | s) on the grid, (subtypes, z), with no floor: a subtype absent from the part, or with no redshift
    mass, is a row of zeros."""
    grid, widths = np.asarray(grid, dtype=np.float64), np.asarray(widths, dtype=np.float64)
    fraction = np.array([float(part["class_fraction"].get(name, 0.0)) for name in names])
    edges = np.asarray(part["redshift_edges"], dtype=np.float64)
    bins = np.clip(np.searchsorted(edges, grid, side="right") - 1, 0, len(edges) - 2)
    inside = (grid >= edges[0]) & (grid <= edges[-1])
    rows = [part["redshift_density"].get(name, np.zeros(len(edges) - 1)) for name in names]
    n = np.stack([np.where(inside, np.asarray(row, dtype=np.float64)[bins], 0.0) for row in rows])
    if np.any(n < 0) or np.any(fraction < 0) or not fraction.sum() > 0:
        raise ValueError("priors.json has a negative density or fraction, or no class mass, in a prior; " + DAMAGED)
    mass = (n * widths).sum(axis=1, keepdims=True)
    n = np.divide(n, mass, out=np.zeros_like(n), where=mass > 0)
    return fraction[:, None] / fraction.sum() * n


def host_redshift_log_prior(grid, widths, measure, redshift, error, outlier=HOST_OUTLIER):
    """The change of prior (`change_prior`) for a host redshift N(redshift, error²), (z,).

    It is the host redshift's probability in each redshift cell (as `cell_widths` draws them), mixed with a share
    ``outlier`` spread over the cells in proportion to their widths, over the cell's prior measure: its density, by
    which the posterior is multiplied. With ``outlier`` 0 it is the Gaussian alone.
    """
    import torch

    if not (np.isfinite(redshift) and np.isfinite(error) and error > 0.0):
        raise ValueError("host_redshift must be finite and host_redshift_error positive")
    edges = torch.from_numpy(_cell_edges(grid))
    low, high = (edges[:-1] - redshift) / error, (edges[1:] - redshift) / error
    # log(Φ(high) − Φ(low)), accurate in both tails
    flip = low > 0
    low, high = torch.where(flip, -high, low), torch.where(flip, -low, high)
    upper = torch.special.log_ndtr(high)
    mass = (upper + torch.log1p(-torch.exp(torch.special.log_ndtr(low) - upper))).numpy()
    widths = np.asarray(widths, dtype=np.float64)
    with np.errstate(divide="ignore"):
        mixed = np.logaddexp(np.log1p(-outlier) + mass, np.log(outlier) + np.log(widths / widths.sum()))
    return mixed - _log_measure(widths, measure)


def redshift_range_log_prior(grid, low, high):
    """The change of prior (`change_prior`) for a redshift between ``low`` and ``high``, (z,): the log of the share
    of each redshift cell inside the range, -inf outside it."""
    with np.errstate(divide="ignore"):
        return np.log(_cell_shares(_cell_edges(grid), low, high))


def redshift_density_log_prior(widths, measure, density):
    """The change of prior (`change_prior`) for a redshift probability density given at each grid point, (z,).

    As for a host redshift (`host_redshift_log_prior`), it is the density's probability in each cell, the density
    times the cell width, over the cell's prior measure.
    """
    with np.errstate(divide="ignore"):
        return (
            np.log(np.asarray(density, dtype=np.float64))
            + _log_measure(widths, "flat_z")
            - _log_measure(widths, measure)
        )


def change_prior(log_posterior, log_ratio):
    """The log posterior under another prior, given log p_new(c, z) / p_old(c, z)."""
    return _normalise(log_posterior + log_ratio)


def _log_measure(widths, measure):
    """The log prior measure of each redshift cell: its width for ``flat_z``, equal for ``flat_log1p``."""
    widths = np.asarray(widths, dtype=np.float64)
    return np.log(np.maximum(widths, 1e-12)) if measure == "flat_z" else np.zeros_like(widths)


def _normalise(log_values):
    peak = log_values.max()
    return log_values - (peak + np.log(np.exp(log_values - peak).sum()))


def _normalise_rows(log_values):
    """Each row normalised, NaN where a row has no probability."""
    peak = log_values.max(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        return log_values - (peak + np.log(np.exp(log_values - peak).sum(axis=1, keepdims=True)))


def _within(grid, widths, bounds):
    """Each cell's centre and width counting only its part between ``bounds`` (low, high), so the summaries of a
    posterior restricted to a redshift range stay inside it; cells wholly outside keep their own."""
    edges = _cell_edges(grid)
    low, high = np.maximum(edges[:-1], bounds[0]), np.minimum(edges[1:], bounds[1])
    inside = high > low
    return (
        np.where(inside, 0.5 * (low + high), np.asarray(grid, dtype=np.float64)),
        np.where(inside, high - low, np.asarray(widths, dtype=np.float64)),
    )


def quantile(grid, probability, value, bounds=None):
    """A quantile of a posterior given as probability masses on an ordered grid, interpolated linearly between
    the midpoint cumulative masses of the cells that hold probability; with ``bounds`` (a redshift range), at the
    centres of the cells' parts inside it."""
    mass = np.asarray(probability, dtype=np.float64)
    mass = mass / mass.sum()
    grid = np.asarray(grid, dtype=np.float64)
    if bounds is not None:
        grid = _within(grid, np.zeros_like(grid), bounds)[0]
    keep = mass > 0.0  # cells without probability (outside a range, or untrained for the classes allowed)
    grid, mass = grid[keep], mass[keep]
    return float(np.interp(value, np.cumsum(mass) - 0.5 * mass, grid, left=grid[0], right=grid[-1]))


def central_interval(grid, probability, mass=0.68, bounds=None):
    """The equal-tailed interval holding ``mass`` of the posterior, inside ``bounds`` if given."""
    tail = 0.5 * (1.0 - mass)
    return quantile(grid, probability, tail, bounds), quantile(grid, probability, 1.0 - tail, bounds)


def window_probability(grid, probability, half=WINDOW, at=None):
    """The probability within ``half`` of each redshift of the grid, or of the redshifts ``at``,
    P(|z' - z| <= half), each cell counted by the share of it inside the window."""
    centres = np.asarray(grid if at is None else at, dtype=np.float64)
    share = _cell_shares(_cell_edges(grid), centres[..., None] - half, centres[..., None] + half)
    return share @ np.asarray(probability, dtype=np.float64)


def redshift_peaks(grid, probability, widths, half=WINDOW, minimum=0.10, dip=0.5):
    """The distinct peaks of a redshift posterior, the main one first and the others by probability, each as
    (redshift, probability).

    The main peak is where the posterior within ``half`` (`window_probability`) is highest. Another counts if that
    is at least ``minimum``, it falls below ``dip`` times its own between it and every peak as high, and its quoted
    probability is at least ``minimum`` too. Each peak's redshift is the highest point of the posterior density
    among the redshifts within ``half`` of it that hold at least 95% of its window probability, refined by a
    parabola in ln(1 + z); its probability is the posterior within ``half`` of that redshift, so it reads as the
    chance that the true redshift lies that close.
    """
    mass = np.asarray(probability, dtype=np.float64)
    mass = mass / mass.sum()
    grid = np.asarray(grid, dtype=np.float64)
    density = mass / np.asarray(widths, dtype=np.float64)
    near = window_probability(grid, mass, half)
    peaks = np.union1d(_local_peaks(near), [int(np.argmax(near))]).astype(int)
    seen, found = [], []
    for i in sorted(peaks, key=lambda i: -near[i]):
        saddle = max((near[min(i, j) : max(i, j) + 1].min() for j in seen), default=-np.inf)
        seen.append(i)
        if found and not (near[i] >= minimum and saddle < dip * near[i]):
            continue
        inside = np.flatnonzero((np.abs(grid - grid[i]) <= half) & (near >= 0.95 * near[i]))
        redshift = _refined_peak(grid, density, int(inside[np.argmax(density[inside])]))
        quoted = float(window_probability(grid, mass, half, at=redshift))
        if not found or quoted >= minimum:
            found.append((redshift, quoted))
    return [found[0], *sorted(found[1:], key=lambda peak: -peak[1])]


def _refined_peak(grid, density, index):
    """The redshift of the density's highest point at ``index``, refined by the parabola through it and its two
    neighbours in ln(1 + z); the grid point itself at an end of the grid or where the parabola does not curve down."""
    grid = np.asarray(grid, dtype=np.float64)
    if index == 0 or index == len(grid) - 1:
        return float(grid[index])
    x = np.log1p(grid[index - 1 : index + 2])
    y = np.log(np.clip(density[index - 1 : index + 2], 1.0e-300, None))
    curvature, slope, _ = np.polyfit(x, y, 2)
    if not np.isfinite(curvature) or curvature >= 0.0:
        return float(grid[index])
    return float(np.expm1(np.clip(-slope / (2.0 * curvature), x[0], x[-1])))


def class_redshifts(grid, posteriors, widths, bounds=None):
    """The main-peak redshift (`redshift_peaks`) of each class's redshift posterior, (classes, z); NaN for a class
    with no posterior at the redshifts allowed. With ``bounds`` (a redshift range), the cells' parts inside it count,
    and each peak stays inside it."""
    if bounds is not None:
        widths = _within(grid, widths, bounds)[1]
    peaks = np.array(
        [
            redshift_peaks(grid, row, widths)[0][0] if np.isfinite(row).all() and row.sum() > 0.0 else np.nan
            for row in posteriors
        ]
    )
    return peaks if bounds is None else np.clip(peaks, *bounds)


def _local_peaks(values):
    """Indices of local maxima, an end counting if it is higher than its neighbour."""
    interior = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1
    last = len(values) - 1
    ends = [end for end, neighbour in ((0, 1), (last, last - 1)) if values[end] > values[neighbour]]
    return np.union1d(interior, ends).astype(int)


def source_probability(logit, fit):
    """The probability that the visits hold a transient, from the source detector's logit and the fit of
    ``calibration.json`` (an affine map of the logit of its score, the score floored at 1e-7 from 0 and 1)."""
    score = np.clip(_sigmoid(float(logit)), EPSILON, 1.0 - EPSILON)
    return float(_sigmoid(fit["slope"] * np.log(score / (1.0 - score)) + fit["intercept"]))


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -700.0, 700.0)))


def summarise(
    log_joint,
    source_logit,
    grid,
    widths,
    measure,
    prior,
    calibration=None,
    host=None,
    *,
    classes=None,
    class_prior=None,
    redshift_prior=None,
    redshift_bounds=None,
    host_outlier=HOST_OUTLIER,
):
    """The results for one object from its log posterior under the training prior.

    Parameters
    ----------
    log_joint : numpy.ndarray
        (classes, z) the log posterior under the training prior (`fuse`).
    source_logit : float
        The source detector's output.
    grid, widths : numpy.ndarray
        The candidate redshifts and their cell widths.
    measure : str
        The redshift prior measure, ``flat_z`` or ``flat_log1p``.
    prior : dict
        `prior_change` of the package's priors.
    calibration : dict, optional
        A fitted ``calibration.json``; without it there is no source probability.
    host : tuple of float, optional
        A host redshift and its 1-sigma error.
    host_outlier : float
        The share of the host redshift's prior left for its being wrong (`host_redshift_log_prior`).
    classes : numpy.ndarray, optional
        (classes,) bool, the classes allowed; the others get no probability.
    class_prior : numpy.ndarray, optional
        (classes,) how common each class is, in place of the survey's mix.
    redshift_prior : numpy.ndarray, optional
        (z,) the change of prior for a redshift range or density, applied as given.

    Returns
    -------
    dict
        ``class_probabilities`` and ``equal_prior_probabilities`` (classes,); ``redshift_posterior`` (z,), summing to 1;
        ``redshift_peaks`` (`redshift_peaks`), ``redshift``, the main peak's redshift, and ``redshift_probability``,
        the probability within 0.1 of it;
        ``redshift_68`` and ``redshift_95``, the posterior's equal-tailed intervals; ``class_redshift_posterior``
        (classes, z), each class's redshift posterior, and ``class_redshifts`` (`class_redshifts`); ``log_evidence``
        (classes, z), the log posterior less the log training prior, before any other prior; ``source_probability``;
        ``untrained_share``, the probability on survey subtypes missing from classes trained at the redshift;
        ``trained_here`` (classes,), the classes with training transients at the redshift; and ``flags``:
        ``several_redshift_peaks``, ``weak_source``, ``untrained_subtypes`` and ``few_classes_trained`` (fewer than
        half the classes trained at the redshift).

    Raises
    ------
    ValueError
        If the classes and redshifts allowed have no probability.
    """
    trained = np.isfinite(prior["training"])
    log_evidence = np.where(trained, log_joint - np.where(trained, prior["training"], 0.0), -np.inf)
    if host is not None:
        log_joint = log_joint + host_redshift_log_prior(grid, widths, measure, *host, host_outlier)
    if redshift_prior is not None:
        log_joint = log_joint + redshift_prior
    survey, equal = prior["survey"], prior["equal_classes"]
    with np.errstate(divide="ignore"):
        if class_prior is not None:
            survey = equal + np.log(class_prior / class_prior.sum())[:, None]
        if classes is not None:
            survey, equal = survey + np.log(classes)[:, None], equal + np.log(classes)[:, None]
    with np.errstate(invalid="ignore"):
        log_survey = change_prior(log_joint, survey)
    if not np.isfinite(log_survey).any():
        raise ValueError("no probability is left within the classes and redshifts allowed; allow more of them")
    joint = np.exp(log_survey)
    equal = np.exp(change_prior(log_joint, equal))
    redshift = joint.sum(axis=0)
    if redshift_bounds is None:
        peaks = redshift_peaks(grid, redshift, widths)
    else:  # the summaries of a posterior restricted to a range stay inside it
        peaks = redshift_peaks(grid, redshift, _within(grid, widths, redshift_bounds)[1])
        peaks = [(float(np.clip(z, *redshift_bounds)), p) for z, p in peaks]
    class_redshift = np.exp(_normalise_rows(log_joint + prior["survey"]))
    untrained = float((joint * np.where(trained, prior["untrained"], 0.0)).sum())
    trained_here = trained[:, int(np.clip(np.searchsorted(_cell_edges(grid), peaks[0][0]) - 1, 0, len(grid) - 1))]
    source = None if calibration is None else source_probability(source_logit, calibration["source_probability"])
    return {
        "class_probabilities": joint.sum(axis=1),
        "equal_prior_probabilities": equal.sum(axis=1),
        "redshift_posterior": redshift,
        "redshift_peaks": peaks,
        "redshift": peaks[0][0],
        "redshift_probability": peaks[0][1],
        "redshift_68": central_interval(grid, redshift, 0.68, redshift_bounds),
        "redshift_95": central_interval(grid, redshift, 0.95, redshift_bounds),
        "redshift_range": redshift_bounds,
        "class_redshift_posterior": class_redshift,
        "class_redshifts": class_redshifts(grid, class_redshift, widths, redshift_bounds),
        "log_evidence": log_evidence,
        "source_probability": source,
        "untrained_share": untrained,
        "trained_here": trained_here,
        "flags": {
            "several_redshift_peaks": len(peaks) > 1,
            "weak_source": None if source is None else source < calibration["source_probability"]["weak_source_below"],
            "untrained_subtypes": untrained > UNTRAINED_LIMIT,
            "few_classes_trained": bool(trained_here.sum() < 0.5 * len(trained_here)),
        },
    }
