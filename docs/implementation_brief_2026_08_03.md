# STRIDER implementation brief — 2026-08-03

> **Historical research record:** This page preserves an earlier experiment or
> decision. It does not define the current STRIDER tool. See
> [`architecture.md`](architecture.md) for the current design.

Written for implementation. The long research log is `research_changeset_2026_08_03.md`
(605 lines, contains retractions and voided claims — read the VERIFICATION STATUS
table at its top before using anything from it). **This file supersedes it for
what to build.**

Scope note: **v2 is frozen and is not being modified.** Its results below z~1.4
are clean and stand. Everything below is v3. v2 appears here only as the source
of measurements, because it is the only trained model that exists.

---

## 1. Three bug fixes

### 1.1 `atlas/build.py:55-100` — per-class phase gate

Currently gates feature windows at rest phase `-20/+80`. That number is the
Sundial paper's cut on *photometric observations entering the SALT3 light-curve
fit* — unrelated to spectral window extraction.

Measured over all 90 TRANSIENTS HEAD/SPEC pairs (776,015 spectra):

| family | generated range | kept by -20/+80 |
|---|---|---:|
| SNIaMODEL00 (Ia) | **-20.0 to +50.0** | 100% |
| NONIaMODEL06 (CCSN) | -86.7 to +279.8 | 51.4% |
| all | | **62.2%** |

Ia stop at exactly +50.0 because `GENMODEL: SALT3.P22-NIR` is only defined there
— a model boundary, not a cut. The sim requested `GENRANGE_TREST: -30.0 100`.

**Change:** per-class phase range derived from each model's actual generated
range. Ia = -20/+50.
**Verify:** per-class bank occupancy; non-Ia window counts roughly double.

### 1.2 `atlas/build.py:332` — `_unit_profile` absolute threshold

```python
if norm <= 1e-8: return result      # returns ZEROS, silently
```

Physical flux is ~1e-19 to 1e-22 erg/s/cm2/A, so **every window of
un-normalised input is silently zeroed and marked unusable** — no exception, no
warning. This cost two full runs on 2026-08-03 alone, and it re-arms itself
whenever upstream normalisation changes.

**Change:** make the threshold relative to the input scale, or raise on an
all-zero result instead of returning one.
**Verify:** feeding raw physical flux must fail loudly, not silently.

### 1.3 `model/strider.py:164` — epoch combination exponent

```python
spectral_logits = (visit_logits*visit_mask).sum(1) / visit_mask.sum(1).clamp_min(1.0)
```

This is a **mean**: 12 visits carry the same logit scale as 1, and nothing
downstream restores N-dependence (`visit_count_reference` appears only in
`evidence_sufficiency`, a separate branch). v2 has the opposite bug — its
`combiner.py:57` returns a bare `sum`, linear in N.

Both are points on one family: `logit = T * N^(-delta) * sum_t w_t s_t`, with
alpha = 1 - delta. v2 is alpha=1, v3 is alpha=0.

**Measured on 22,012 objects** (v2 eval, global temperature refit at each alpha so
overall 68% coverage is held fixed, then alpha chosen to flatten coverage vs N):

| alpha | what it is | coverage spread across epoch count |
|---:|---|---:|
| 1.00 | v2 as built | **15.2%** (77.2% -> 61.9%) |
| **0.75** | **measured optimum** | **3.6%** |
| 0.50 | sqrt(N) | **17.8%** (58.4% -> 76.2%) — worse than doing nothing |

Radar noncoherent-integration theory predicts alpha in 0.7-0.9 for N in 2-100
(Richards); sqrt(N) is a folk simplification, not the result. The empirical fit
and the theory agree.

**Change:** `alpha = 0.75` as a fitted, class-independent scalar.
**Verify:** 68% and 95% credible-interval coverage **flat in epoch count**. This
is the acceptance test and it is measurable on data that already exists.

**Caveat to record:** the weights come from `_ivar_weight_channel`, which
median-centres ivar per object — its own docstring says this "kills any
absolute-S/N signal". Any inverse-variance normalisation is dimensionless
bookkeeping until that centring is undone. Also, alpha=0.75 may reflect
*epoch correlation* (shared host subtraction, calibration, template mismatch,
giving sqrt(N/(1+(N-1)rho))) rather than noncoherent integration. Same fix,
different meaning; measure rho rather than assuming.

---

## 2. Data-loader change — fixes the largest leak at source

**Truncate every class at +50 rest days.**

Ia stop at +50 by SALT3 construction; other families run to +280. Consequence:

| | epochs (median) | observer span (median) |
|---|---:|---:|
| Ia | 20 | 131 d |
| CCSN | 36 | 268 d |

A single scalar with **zero spectral content** then classifies Ia vs CCSN:

| feature | accuracy |
|---|---:|
| observer-time **span** alone | **90.5%** |
| early->late gap alone | 88.5% |

This is the largest single defect found. It is a **simulation artefact** (real Ia
are observed into the nebular phase; SALT3's template stops, nature does not), so
a model that learns it will degrade on real data. Fixing it in the loader costs
zero modelling capability — do this before any architectural work.

**Verify:** a classifier on **timing features alone** (epoch count, span,
inter-visit gaps, no flux) must sit at **chance**. Until it does, no
classification number is admissible.

**Also measured, and still open:** gold-Ia selection rate depends on epoch count
even within fixed z bins — at p(Ia)>0.99, z 1.6-2.2: 61% selected with <20
epochs, 99.7% with >=20 (rho=+0.56). Contamination is 0.0% throughout, so this is
a **selection-function** problem, not a purity problem — but it is a
redshift-dependent one, which matters for cosmology. Re-measure after the
truncation fix.

---

## 3. Input representation — coadd

**Coadd all retained epochs per object, observed frame, per-bin inverse-variance
weighted.** No per-epoch renormalisation before coadding (it destroys the light
curve; production's per-visit `background_scale` division is the correct
normalisation and should be kept).

Two independent justifications:

**S/N.** ~1.8x over the best single epoch, ~4.9x over the median epoch.

**Leak immunity — the important one.** Four-arm transplant protocol at z 1.35-2.0,
the band where v2 fails. Positive control passed (observed arm, median |dz| =
0.0567).

| | v2 (z 1.68-2.0) | **coadd + ONIR scan (z 1.35-2.0)** |
|---|---:|---:|
| signal + transplanted envelope -> follows SIGNAL | **35%** | **88%** |
| noise alone -> reports target z | **68%** | **3%** |

v2 follows a transplanted noise envelope more often than the signal. The coadd
scan never does — variance-source lock 0% in every arm. Mechanism: source adds
coherently across epochs (gain N), noise incoherently (gain N^alpha, alpha<1), so
signal-to-leak improves with N. Radar: coadd-then-match strictly dominates
match-then-sum.

**Caveat:** this was measured with untrained template matching, so it validates
the REPRESENTATION. A trained coadd model could still learn the shortcut from
labels — see the NERSC test in section 6.

---

## 4. Model configuration

| element | setting | evidence |
|---|---|---|
| features | 15 named ONIR anchors, local windows | beats global cross-correlation ~100x at z~1 |
| redshift | **integer shift** on log-lambda grid | no per-trial interpolation; 200 km/s bins |
| **scan range** | **z in [0.05, 2.0]**, abstain beyond | Si II 6355 exits the prism at **z=1.90** |
| phase axis | **none, for now** | see 6.2 — this is an experiment, not a settled decision |
| epoch combination | alpha = 0.75 | section 1.3 |
| sigma | **precision term only, never a feature** | see below |
| atlas gate | per class; Ia -20/+50 | section 1.1 |

**On the error array.** Write sigma^2 = A(lambda) + B(lambda)*flux(lambda). The
only place the source enters is the factor `flux`, which the network already has
at far better S/N on the flux channel. Measured: d(log FLAMERR)/d(log SIM_FLAM) =
0.0055 per bin across epochs — the source is ~1% of the variance. So restricting
sigma to a precision role (weighting, masking, likelihood denominator) costs
approximately **nothing** and removes the channel that produced the leak.

**Do not** whiten by dividing flux by the per-event sigma array. It stamps
sigma's structure multiplicatively into the signal channel where no downstream
check can see it, and it destroys the transplant diagnostic that proves the leak
is closed. If whitening is wanted, use a smooth model sigma or a population
average.

---

## 5. What NOT to build

These were considered and rejected on evidence, not taste:

- **Learned temporal encoder / Set Transformer / attention-MIL pooling.** Trajectory
  channel SNR is ~1.0-1.5 (~1 nat/object); attention over epochs is also a
  learnable path to read cardinality, which is the 90.5% leak. Set Transformer's
  PMA reads set size directly.
- **sqrt(N) scaling (alpha=0.5).** Measured worse than v2's alpha=1 (spread 17.8%
  vs 15.2%).
- **Chernoff fusion / covariance intersection** for combining channels. Its fused
  covariance does not depend on the disagreement between channel means at all, so
  it *cannot* broaden on disagreement; and being a convex combination of
  informations it can never exceed the better single channel. Use an
  epsilon-contaminated likelihood instead if disagreement-broadening is wanted.
- **Multi-lag / temporal filter bank.** Deviation matrix effective rank 1.27; a
  single early-vs-late contrast captures ~92%. (Caveat: measured on mean-removed,
  shape-normalised, Ia-only data as a median — treat as a default, not a
  prohibition.)

---

## 6. Tests, in order

### 6.1 Acceptance gates — each must pass before the next means anything

| gate | criterion | status |
|---|---|---|
| **Cardinality** | timing-features-only classifier at **chance** | currently **90.5%** — FAILS |
| **N-scaling** | 68%/95% coverage **flat in epoch count** | currently spread 15.2% (v2) |
| **Leak** | four-arm transplant vs the **trained** coadd model: variance-source lock at chance | untrained version passed (0%) |

### 6.2 Open experiments

**t0-accuracy curve.** Phase conditioning with a *correct* t0 gives a large,
highly significant gain (C vs B: -6.82 pp outliers p=0.0041; median 0.0059 vs
0.0103, Wilcoxon p=9.9e-06). A crude proxy (first epoch + fixed offset) destroys
it entirely (D vs C: +6.36 pp, p=0.0094). So **t0 accuracy is the lever, not
phase per se.** Run proxy / SALT3 `PKMJD` / truth as three points. This decides
whether the phase axis comes back.

**Classification with a working control.** The earlier result ("temporal evolution
classifies better") is **VOID** — the N_FIX=8 control subsampled uniform in rank,
not phase, so Ia spanned 64.8 d and CCSN 135.5 d, and span-alone beat the claimed
spectral result in every z bin. Rebuild with a fixed rest-phase grid, add a
timing-only arm, and re-test after the section-2 truncation fix.

**Coadd performance numbers.** Also **VOID** — the "inverse-variance" coadd applied
`1/sigma^2` to already-rescaled flux, making weights wrong by ~21x median. Rerun
with `1/(s_i*sigma_i)^2`, or simply do not renormalise per epoch.

---

## 7. Order of work

1. Section 1 bug fixes (hours)
2. Section 2 loader truncation + timing-only gate (the largest leak, fixed at source)
3. Section 3 coadd input representation
4. Re-run the voided tests from 6.2 with working controls
5. NERSC: trained coadd model vs the four-arm transplant; 15-class and non-Ia
   (everything measured here is Ia-only)

---

## 8. Standing practice

Every diagnostic gets a **positive control that gates its output** — if the
control fails, the script exits rather than printing a table. Four scripts failed
silently on 2026-08-03; three were caught by luck. The one that had a gate caught
its own failure immediately.

Every claim touching sigma or the coadd carries a **source-free null test**
(same pipeline, flux replaced by noise from the same envelope), and the null is
**re-run, never inherited**, when the model or error handling changes. Use a
permutation baseline, not a uniform-grid one — the scan does not produce uniform
argmax under noise.
