# STRIDER research change set — 2026-08-03

> **Historical research record:** This page preserves an earlier experiment or
> decision. It does not define the current STRIDER tool. See
> [`architecture.md`](architecture.md) for the current design.

Five items from the phase/timing audit. Items 1–3 are code changes; item 4 is a
data-generation change; item 5 is an experiment specification, not a code change.

The general format and flow of v3 is settled and is not up for revision here.

---

## 1. Atlas phase gate must be per-class, not global

**Finding.** `atlas/build.py:55-100` gates feature windows at rest phase
`-20/+80`. That number matches the Sundial paper's cut on *photometric
observations entering the SALT3 light-curve fit* (paper §5, Table 4). It has no
bearing on spectral window extraction — the two are unrelated procedures.

**Evidence.** Measured across all 90 TRANSIENTS HEAD/SPEC pairs, 776,015 spectra:

| family | generated rest-phase range | kept by `-20/+80` | discarded |
|---|---|---:|---:|
| SNIaMODEL00 (Ia) | **-20.0 to +50.0** | 100% | 0% |
| NONIaMODEL06 (CCSN) | -86.7 to +279.8 | 51.4% | 47.4% |
| NONIaMODEL08 | — | 50.5% | 46.3% |
| NONIaMODEL01 | — | 27.5% | 68.4% |
| **all** | | **62.2%** | **36.8%** |

The Ia ceiling at exactly +50.0 is a **model boundary, not a cut**. The sim
requested `GENRANGE_TREST: -30.0 100` for Ia
(`1_SIM/TRANSIENTS/PIP_JP_HOURGLASS2_FIXED_TRANSIENTS.input`), but
`GENMODEL: SALT3.P22-NIR` is only defined over -20 to +50, so SNANA cannot
generate outside it. No amount of additional data will fill that range.

The global gate is therefore wrong in both directions at once: it leaves 30 days
of the Ia phase axis permanently unpopulated, and discards 37% of non-Ia spectra
for no reason.

**Change.** Replace the global gate with a per-class phase range, derived from
each model's actual generated range rather than hardcoded. For Ia this is
-20/+50 by SALT3 construction.

**Verify.** After the change, no class should have an empty or near-empty phase
cell attributable to the gate; report per-class occupancy. Non-Ia window counts
should rise substantially (roughly 2x for CCSN).

---

## 2. Cadence augmentation — epoch count and observer span are a class shortcut

**Finding.** The Ia phase ceiling creates a time-series shape that separates
classes with no spectroscopy at all.

**Evidence.** Per-object, all TRANSIENTS files:

| family | n_obj | epochs (median) | observer span (median) | frac with any epoch > +50 d |
|---|---:|---:|---:|---:|
| **SNIaMODEL00 (Ia)** | 8,218 | **20** | **131.5 d** | **0.0%** |
| NONIaMODEL06 (CCSN) | 14,837 | 36 | 268.2 d | **82.2%** |
| NONIaMODEL08 | 564 | 45 | 321.7 d | 80.9% |
| NONIaMODEL07 | 573 | 28 | 187.1 d | 76.6% |

Zero of 8,218 Ia have any epoch past +50 days; 82% of CCSN do. Epoch count and
observer span differ by roughly 2x between Ia and CCSN, and both are readable
from MJD alone.

This is consistent with the pilot's own controls
(`docs/local_pilot_results_2026_08_02.md`): zeroing all times collapses accuracy
0.926 -> 0.512 (chance for binary), while reassigning times *within* an object
costs only 0.926 -> 0.886. The information is in the **set** of visit times, not
in the spectrum-to-time pairing — which is exactly what epoch count and span are.

The effect is partly artificial. Real Type Ia are observed well past +50 days
into the nebular phase; it is SALT3's template boundary that stops at +50, not
nature. The simulation sharpens a soft, genuine physical difference into a hard
and exploitable one, so a model that learns it will degrade on real data.

**Change.** Add a training-time cadence augmentation that randomly truncates and
subsamples each object's epoch list so that **epoch count and observer-time span
distributions overlap across classes**. No spectrum is discarded from the store;
this is a per-batch sampling operation. Cost at deployment is zero.

**Verify.** After augmentation, a classifier trained on **timing features alone**
(epoch count, span, inter-visit gaps — no flux) should sit near chance. That is
the acceptance test. Run it before and after; the "before" number is the size of
the leak.

---

## 3. Window-local detrending before the NCC

**Finding.** v3 uses a log-wavelength grid throughout (`encoded_onir.py:393`
`_uniform_log_step` enforces uniformity), which is correct and matches the field.
But there is **no continuum handling anywhere** in `src/strider/` — the only
normalization is `normalize_valid_bins` (`encoded_onir.py:383-390`), a
whole-spectrum z-score. That removes offset and global scale but leaves the
continuum *shape* intact.

Every comparable tool — SNID, DASH, ABC-SN, SNID-SAGE — divides out the continuum
before matching.

**Change, deliberately narrower than the field standard.** Do **not** add global
continuum division. Continuum shape carries real class information (SLSN blue
continua, reddened Ia), and Roman prism flux calibration is far better than the
heterogeneous archival spectra SNID was built to handle, so the original
motivation is weaker here.

Instead, **remove a linear fit across each ONIR window before computing the
NCC**. Within a ~15-bin window the continuum is close to linear, and a residual
slope correlates with the prototype and dilutes the match. This targets the
actual failure mode, costs almost nothing, and leaves global continuum shape
available to the CNN branch.

**Verify.** NCC peak sharpness and redshift-scan contrast should improve. Watch
for degradation in classes whose discriminant is partly continuum slope.

---

## 4. Photo-z prior must be synthesized, and must never be the headline

**Finding.** `HOSTGAL_PHOTOZ` is populated for **0 of 82,177** objects across all
90 TRANSIENTS files. `HOSTGAL_SPECZ` is populated for 97.2%. Any photo-z prior
work therefore requires synthesizing the prior.

**Why the prior cannot replace the scan.** For Si II 6355, redshift uncertainty
translates to wavelength smear as a fixed multiple of the feature width,
independent of z (both scale as 1+z):

| | sigma_z | smear | as fraction of feature width |
|---|---:|---:|---:|
| spec-z | 0.001 | 10-18 A | 0.04x |
| photo-z, optimistic | 0.02(1+z) | 191-356 A | 0.8x |
| photo-z, typical | 0.05(1+z) | 477-890 A | **2.1x** |

ABC-SN's de-redshift-then-classify approach requires a **spec-z**. A photo-z is
20-50x too coarse for it — it smears Si II across twice its own width. The
redshift scan is therefore not primarily a redshift product; it is the mechanism
that makes classification work without a spec-z. As a *prior over* the scan,
however, sigma = 0.02-0.05 against a scan range of 0-3 is a 10-30x reduction in
search space and suppresses alias solutions.

**Change.** Synthesize photo-z as truth z perturbed by a declared error model
plus a declared catastrophic-outlier fraction. The outlier rate is the parameter
that matters most; make it a config field, not a constant.

**Discipline — this is the important part.** A synthesized photo-z built from
truth z carries truth information more precise than STRIDER's own spectral
estimate over much of the range. Without care the model learns to read the prior
and ignore the spectra, and every "spectral" number silently becomes
prior-driven. Therefore: **the no-prior arm is always the headline result**; the
prior arm is reported separately as the assisted result. Apply the prior
ambiguity-gated, not unconditionally.

---

## 5. Phase axis — an experiment arm, not a decision

Do not restore the ONIR phase axis on the strength of argument. The prior art is
genuinely split:

- **SNID / SNID-SAGE**: phase is a template axis (type x age), determined by
  matching. Never supplied.
- **DASH**: phase is an output axis — ~306 classes as (type, age-bin) pairs.
- **ABC-SN (2025)**: phase **dropped entirely**, and it outperforms DASH on
  nearly all classes.

ABC-SN is the most recent and wins, which points away from a phase axis — but
DASH split 17 types x 18 age bins, so ABC-SN's gain may come from coarser class
granularity rather than from phase being uninformative. That confound is not
resolvable from the literature.

**Therefore.** Keep v3 **phase-neutral as the default**. Run phase-conditioned as
an arm on NERSC where the flat-z statistics make the comparison real, with the
phase offset **scanned rather than supplied** so the arm stays deployable. Decide
on class metrics at >= 5 seeds.

Note that every tool above classifies a **single spectrum** with no access to
observation timing. STRIDER's use of a time series is a genuine extension, but it
means there is no prior art protecting us from item 2 — which is why the cadence
augmentation is the urgent item and the phase axis is not.

---

## Order

1. Item 1 (atlas phase gate) — smallest, unblocks non-Ia bank occupancy.
2. Item 2 (cadence augmentation) — highest scientific priority; blocks any
   claim that classification gains are spectral.
3. Item 3 (window detrending) — cheap, testable locally.
4. Item 4 (photo-z synthesis) — needed before any assisted-result claim.
5. Item 5 — NERSC, after 1-3 land.

---
---

# Part 2 — measured results, same day

Raw-flux normalised cross-correlation against ONIR feature windows. **No training,
no learned representation** — so these are lower bounds and mechanism
demonstrations, not benchmarks.

> **Correction:** an earlier version of this line claimed the scripts use
> STRIDER's own `feature_geometry` / `align_to_rest_grid` /
> `extract_feature_windows`. Only the first script did. The rest re-implement the
> geometry, and several import nothing from `strider`. Resampling was verified
> to match production to 2.1e-16, but **normalisation diverges**: production
> divides each visit by `quantile(FLAMERR>0, 0.25)` (`data/prepare.py:194`) and
> retains a 3.4x per-epoch amplitude range that these scripts flatten to 1.0.
> That cancels for per-window cosine arms; it matters ONLY for coadds.

## VERIFICATION STATUS — read before acting on anything below

Independently reviewed the same day (implementation diff + statistics).

| item | status |
|---|---|
| 6 — envelope z<=2 (Si II exits 1.90) | **SOLID** — optics, no script involved |
| 1 — atlas gate; Ia span -20/+50 | **SOLID** — FITS metadata |
| 2 — timing shortcut, 0% vs 82% | **SOLID** — FITS metadata. See the addendum: observer-time SPAN ALONE classifies at **90.5%** |
| 14 — v2 sums / v3 means visit logits | **SOLID** — code read |
| 5/13 — temporal decomposition 80.1/31.9/3.5 | **VERIFIED** — 0/927 ceiling violations; plus rank-1 finding |
| 11 — `_unit_profile` absolute threshold | **SOLID** — cost two runs on 2026-08-03 alone |
| leak boundary z~1.4 for v2 | **SOLID** — production path, three independent lines |
| **coadd suppresses the leak** | **SOLID** — control-gated, see addendum |
| 9 — coadd/IV weighting performance | **VOID, rerun required** — the "inverse-variance" coadd was mis-weighted (`1/sigma^2` applied to already-rescaled flux; weights wrong by 21x median). The 0.0099 number recurs but is NOT attributable to IV weighting |
| 12 — temporal evolution classifies | **VOID** — the N_FIX=8 control failed. Ia span 64.8d vs CCSN 135.5d; span ALONE gives 90.5%, beating the claimed 75.5% spectral result in every z bin |
| 13 — accumulation curve | **PARTIAL** — knee at 12 holds on a fixed cohort; "coadd halves outliers" is p=0.09, direction only; the k=24 point is a different (higher-z) population |
| null test | **CONCLUSION HOLDS, evidence restated** — the uniform-on-grid baseline was the wrong null. Permutation baseline: 10.9% vs 10.8%, p=0.51 (0.06 sigma). Excludes leaks >5.5 pp only; 1 pp needs n~5900. It also cannot be repeated reliably as written (`hash()` on np.str_ is per-process randomised) |

Method notes that make them leak-resistant by construction:

- `FLAM` values only. `FLAMERR` enters solely as a validity mask (`err > 0`); its
  magnitude is never used. The variance-envelope channel cannot contribute.
- Banks from FITS blocks 1-3, tests from blocks 9-10. No object overlap.
- Banks built from clean `SIM_FLAM`; tests on noisy `FLAM`.
- Every window is mean-subtracted and unit-normalised before matching, so
  absolute flux scale and absolute noise level are both discarded.
- Redshift is an **integer shift** on a 4x-oversampled log-lambda grid
  (200.6 km/s per bin, dz = 0.0013 at z=1). No per-trial interpolation.
- No continuum fitting of any kind.

## 6. Operating envelope is z in [0.05, 2.0], set by optics not convenience

Redshift at which each ONIR feature leaves a 7450-18432 A window:

| feature | rest A | exits at z |
|---|---:|---:|
| Ca II NIR | 8579 | 1.15 |
| O I 7774 | 7774 | 1.37 |
| He I 7065 | 7065 | 1.61 |
| H alpha 6563 | 6563 | 1.81 |
| **Si II 6355** | 6355 | **1.90** |
| Si II 5972 | 5972 | 2.09 |
| Fe/Ca blue complexes | 3945-5620 | 2.3-3.7 |

Si II 6355 — the Ia identity anchor — exits at **z = 1.90**. Past that only the
blue Fe/Ca complexes remain, and those are shared with other classes. The
measured performance cliff sits exactly there.

**Change.** Set the scan range to z <= 2.0 and treat anything beyond as an
abstention, not a measurement.

## 7. Restricting the scan range removes the high-z collapse

The high-z failures were **aliases, not noise**: outlier ratios
(1+z_rec)/(1+z_true) cluster near 1.2, which is the spacing of several ONIR
feature pairs (He/Na 5885 / Fe/Mg 4861 = 1.211; H alpha 6563 / S II 5454 = 1.203;
O I 7774 / Si II 6355 = 1.223). The scan was locking onto the feature comb
shifted by one feature, piling up near z ~ 2.5.

Cutting the grid at z = 2 eliminates them:

| z bin | flat coadd, scan to z=3 | flat coadd, scan to z=2 |
|---|---:|---:|
| 1.5-1.75 | 0.7375 | **0.0249** |
| 1.75-2.0 | 0.4481 | **0.0475** |

## 8. Phase conditioning does NOT survive a deployable peak time

220 test Ia, z <= 2, four arms. C uses truth `SIM_PEAKMJD`; D uses first-epoch
MJD plus a -18.3 rest-day calibration offset fitted on training data (fully
deployable, no truth).

| arm | median abs dz | outliers >0.1 |
|---|---:|---:|
| A  best single epoch | 0.0689 | 49% |
| B  flat coadd, no phase | **0.0103** | **18%** |
| C  phase stacks, **truth** t0 | 0.0059 | 11% |
| D  phase stacks, **proxy** t0 | 0.0108 | 17% |

**REFRAMED after statistical review.** "D is indistinguishable from B" is
underpowered as stated — paired TOST gives equivalence only within +-4.9 pp, so
a +-2 pp claim would need n=1076. But the informative contrast is against C, and
there the result is decisive:

| contrast | outliers | p | median abs dz | Wilcoxon p |
|---|---:|---:|---|---:|
| **C (truth t0) vs B (no phase)** | **-6.82 pp** | **0.0041** | 0.0059 vs 0.0103 | **9.9e-06** |
| **D (proxy t0) vs C** | **+6.36 pp** | **0.0094** | 0.0108 vs 0.0059 | **5.6e-09** |

So: **phase conditioning delivers a large, highly significant gain given a
correct t0, and a crude proxy destroys it entirely.** That is a positive
mechanistic finding about *t0 error*, not a null result about phase — and it is
decisive at the effect size that matters, because the +-4.9 pp equivalence bound
sits inside the 6.8 pp benefit C demonstrates.

**Consequence:** t0 accuracy is the lever. SALT3 `PKMJD` (with calibrated
`PKMJDERR`) sits between this crude proxy and truth, so the three-point curve —
proxy / SALT3 / truth — is now a priority experiment rather than a nicety. Do not
drop the phase axis on the strength of arm D alone.

## 9. Coadding is the deployable representation

Uniform coadd over all retained epochs, observed frame (no z needed):

- **1.8x S/N** over the best single epoch, 4.9x over the median epoch.
- Verified not a search-space artifact: holding template count fixed, coadding
  still improves median abs dz by 7.8x and cuts outliers 4x.
- **Weighting: use per-bin inverse variance.** Measured on 220 test Ia at z<2:

  | weighting | median abs dz | outliers >0.1 | null: frac<0.1 (chance 8.6%) |
  |---|---:|---:|---:|
  | uniform | 0.0103 | 18% | 11% |
  | **per-bin 1/sigma^2** | **0.0099** | **13%** | **10%** |

  Inverse variance cuts the outlier rate by 28%. It does *not* show up in
  integrated S/N, which measures 0.99-1.01x — that statistic is insensitive to
  the effect. Per-bin weighting suppresses individual bad bins (sky residuals,
  hot pixels) that throw the cross-correlation onto an alias; outlier rate
  depends on the worst bins, S/N does not.

  **Standing condition:** this arm uses `FLAMERR` magnitude, so unlike the
  flux-only arms it is not leak-immune by construction. Its null test passes
  here, but the null must be **re-run** after any change to the error model and
  for any trained model — never inherited.

  Optimal weighting would be signal/variance, obtainable honestly from the
  observed photometric light curve, but that is a separate declared channel.
- The earlier warning in this document that inverse-variance weighting smuggles
  the source back in was **overstated**: measured per bin across epochs,
  d(log FLAMERR)/d(log SIM_FLAM) = 0.0055, i.e. the source is ~1% of the
  reported variance.

## 10. Restore the v2 atlas design decisions

`strider-v2/analysis/build_onir_atlas.py` made six choices v3 dropped:

| | v2 atlas | v3 |
|---|---|---|
| phase bins | -20,-15,-10,-5,0,5,10,15,20,30,40,60,80 (non-uniform: 5d near peak, 20d late) | none |
| continuum | Gaussian high-pass, sigma=48 bins: `flux - gaussian_filter1d(flux, 48)`, then /std | none |
| window width | velocity-aware **and z-dependent**, from each feature's velocity range and blend wavelengths, x(1+z), 24-137 bins | fixed per feature |
| resolution | degraded to R=100 (Roman prism) before extraction | not done |
| **source** | **SNID-SAGE library — real observed spectra**, 4344 rows | Sundial **simulations** |
| matching space | learned 8-dim subspace (`token_proj: d_model -> scan_dim=8`) | raw profile |

Two matter most:

**v2's continuum treatment is subtractive, not divisive.** `flux - smoothed(flux)`
is a linear filter and cannot blow up where a fitted continuum crosses zero. A
4th-order polynomial *division* destroyed the high-z arm of the first test run
here; the ONIR local-window path (no continuum at all) beat it by ~100x at z~1.
Both are safe; polynomial division is not. If continuum handling is reinstated,
use v2's high-pass.

**v2's signatures came from real spectra.** Given that the entire variance-leak
problem was "the model learned the simulation", grounding the ONIR anchors in
observed supernovae rather than Sundial output is a substantive robustness
property that v3 gave up.

## 11. Bug: `_unit_profile` fails silently on un-normalised input

`atlas/build.py:332` rejects a window with `if norm <= 1e-8: return result` — an
**absolute** threshold. Physical flux is ~1e-19 to 1e-22 erg/s/cm2/A, so every
window of un-normalised input is silently zeroed and marked unusable, with no
exception and no warning. It cost a full run here before being spotted.

**Change.** Make the threshold relative to the input scale, or raise on an
all-zero result rather than returning one.

## 12. Temporal evolution DOES classify — at low redshift

Nearest-class-prototype, Ia vs CCSN, rest frame, **every object subsampled to
exactly 8 epochs** so epoch count and its S/N cannot carry class information.
600 test objects, 1,400 training.

| arm | accuracy | Ia recall | CCSN recall |
|---|---:|---:|---:|
| STATIC (coadd only) | 72.5% | 82.3% | 62.7% |
| **EVOLUTION (late-half minus early-half)** | **75.5%** | **89.0%** | 62.0% |
| **BOTH** | **77.0%** | **90.3%** | 63.7% |

| arm | z 0.3-0.8 | z 0.8-1.3 | z 1.3-2.0 |
|---|---:|---:|---:|
| STATIC | 72.2% | 70.1% | 75.3% |
| **EVOLUTION** | **85.7%** | 71.3% | 74.0% |
| BOTH | **87.2%** | 71.3% | 77.1% |

### RETRACTION — statistical review, same day

The framing above ("evolution beats static") **does not survive**. Corrected:

- **EVOLUTION vs STATIC overall: +3.0 pp, McNemar exact p = 0.215.** Noise.
- Low-z bin is **n=133**, imbalanced **42 Ia / 91 CCSN**, majority baseline 68.4%.
- EVOL vs STATIC at low z: +13.53 pp, p=0.0153 — **fails** Benjamini-Hochberg
  across the 12 tests run (3 arms x 3 z-bins + overall; BH critical 0.0125).
- **Only surviving result: BOTH vs STATIC at low z, +15.04 pp, p=0.00018.**
- Balanced accuracy at low z goes 61.7% -> 87.0%, so the low-z effect is real and
  not a class-prior artefact.

**Confound to fix before re-testing:** the STATIC baseline is *broken* at low z,
not merely worse — Ia recall 33.3%, predicting Ia for 17.3% of objects against a
truth rate of 31.6%. A single global prototype per class with no redshift
conditioning is systematically miscalibrated per z bin. Part of the gap is
"evolution is less sensitive to a broken static baseline", not temporal
information.

**Revised design consequence:** COMBINE static and temporal at low z. Do not
build a temporal channel on the claim that it beats the static spectrum. The
trajectory templates remain free (dS/dphi across existing bank phase knots) and a
learned temporal encoder remains unjustified — but the motivating effect size is
now +15 pp for the combination at z<0.8, not +13.5 pp for evolution alone.

## 13. Evidence accumulation: ~12 epochs needed, saturates at ~16

Epochs accumulated in TIME order, 120 test Ia, z<2.

| epochs | coadd-then-scan | outliers | scan-then-combine | outliers |
|---:|---:|---:|---:|---:|
| 4 | 0.389 | 71% | 0.403 | 75% |
| 6 | 0.153 | 52% | 0.253 | 57% |
| 8 | 0.031 | 42% | 0.025 | 38% |
| **12** | **0.0107** | 21% | 0.0118 | 17% |
| **16** | **0.0092** | **9%** | 0.0100 | 16% |
| 24 | 0.0133 | 9% | 0.0138 | 16% |

Sharp transition between 6 and 12 epochs — the scan either locks on or it does
not. Operationally: a redshift is not usable below ~12 epochs.

**Coadd-then-scan and scan-then-combine tie on the median** (0.0092 vs 0.0100) but
coadding cuts outliers roughly in half (9% vs 16%). So stacking buys robustness
against aliases, not precision. A coherent-integration argument predicting a
median advantage was tested and is WRONG.

## 14. Epoch combination: the legacy and current routes have opposite bugs

- **v2** (`strider/latest/combiner.py:22,57`): defaults to `snr_sum` and returns
  `weighted.sum(dim=1)`; every config sets `combiner_mode: snr_sum`. With bounded
  NCC scores at fixed temperature this scales the effective inverse temperature
  as **N**, so confidence grows with epoch count. The in-code comment claims it
  "preserves sqrt(N)" — the comment is wrong.
- **v3** (`src/strider/model/strider.py:164`): divides by the visit count, i.e. a
  **mean**, so 12 visits carry the same logit scale as 1 and the model cannot
  become more confident with more data through that path.

**Fix**: `Z = sum(w_t s_t) / sqrt(sum(w_t^2))` — the L2 norm of the weight vector,
which gives genuine sqrt(N) and makes the temperature N-independent.

**Structural guarantee to assert, not assume:** if the per-epoch weights have no
class axis, epoch count can move CONFIDENCE but can never move the ARGMAX class.
v3's weights are already `(B,T)` broadcast over class and z — make it an explicit
assertion so it cannot regress.

## Revised order

### The design, as the measurements now specify it

| element | choice | evidence |
|---|---|---|
| input | observed-frame coadd, **per-bin 1/sigma^2** | item 9 |
| second channel | **late-half minus early-half** stack difference | item 12 |
| features | ONIR named windows, log-lambda, z as integer shift | item 7 |
| scan range | **z in [0.05, 2.0]** | item 6 |
| phase axis | **OPEN — pending the t0-accuracy curve** | item 8 (revised) |
| epoch combination | `sum(w s) / sqrt(sum(w^2))` | item 14 |
| sigma | **precision term only, never a feature** | item 4 |
| atlas gate | per class; Ia -20/+50 | item 1 |

Untrained, this configuration measures **median abs dz = 0.0099, 13% outliers**.
That is the floor a trained model must beat.

**Null test, corrected.** The original uniform-on-grid chance baseline was the
wrong null: the scan does not produce uniform argmax under noise, it piles up at
high z (58.2% above z=1.5 vs 17.3% uniform, KS p=2e-30), and since true z is also
high-z skewed this understated chance. Under a proper **permutation null**
(shuffle which object each null prediction pairs with — preserves both marginals,
breaks only the association):

| arm | observed | permutation null | p |
|---|---:|---:|---:|
| B coadd | 10.91% | 10.44% | 0.44 |

Excess **+0.47 pp, CI [-3.25, +4.45]** — 0.14 sigma, not the 1.2 sigma originally
reported. Stronger still, Spearman rho(null prediction, z_true) is NEGATIVE for
every arm (B: -0.117); leakage must be positive, and real data gives **+0.759**.

**State the bound, do not imply zero:** at n=220 this excludes only leaks larger
than **5.5 pp**. Excluding 3 pp needs n=694; 1 pp needs n=5944.

**Rank-1 finding (verified).** Each arm's fraction-of-ceiling is *identically* the
cosine alignment between its temporal contrast and the deviation pattern (verified
to 5.6e-16), so by Cauchy-Schwarz no arm can exceed 100%. Against a tighter
per-contrast ceiling: adjacent **4.1%**, telescope **36.9%**, stacks **92.1%**.
And the deviation matrix is essentially **rank-1** — median effective rank
**1.27**. Shape-normalised spectral evolution is ONE dominant mode with a monotone
time amplitude. That is why a crude half/half contrast captures ~92% of everything
available, and it means **multi-lag / temporal-filter-bank schemes have almost no
headroom**. Do not build one.

### Order

1. **Bug fixes (hours).** Item 1 (atlas gate per class), item 11
   (`_unit_profile` absolute threshold), item 6 (scan range z<=2), item 14
   (v3 visit mean -> L2-normalised sum).
2. **Settle the v2 question (1 day, no new code).** Re-run
   `signal_vs_mismatched_native_variance.py --zlo 0.4 --zhi 2.0` at large n, and
   `noise_channel_discriminator.py` at z<2 where it has never been run. Decides
   whether v2's z<2 results stand, which changes how much of v3 is a rebuild.
3. **Scale the local tests (1 day).** 82,177 objects are available; roughly 1%
   were used. Highest priority is the **null margin** — 10.9% vs 8.6% chance is
   only ~1.2sigma, and every result here rests on it.
4. **Cadence augmentation + acceptance test (days).** Criterion: a classifier on
   timing features ALONE must sit at chance. Until it passes, no classification
   number is defensible, including item 12's +13.5 pp (which was controlled by
   fixing N=8, not in general).
5. **Build the two channels (weeks).** Coadd + trajectory. Trajectory templates
   are free (dS/dphi across existing bank phase knots); no learned temporal
   encoder.
6. **NERSC.** Trained model vs the 0.0099 floor; 15-class and non-Ia (everything
   measured here is Ia-only); item 10 (restore v2 atlas construction); item 4
   (photo-z synthesis, no-prior arm always the headline).

### Out of scope, recorded

Photometric fusion. The Roman light curves are available locally (90 PHOT files,
6 bands, FLUXCAL/FLUXCALERR/MJD) and are the natural home for BRIGHTNESS
evolution, which the spectral path discards by design when it removes absolute
scale. Spectral evolution dies above z~0.8 (item 12) while broadband photometry
is far deeper, so photometry plausibly covers exactly that gap. This is a
follow-up paper, not this one — and it reopens the cadence leak, since light-curve
duration is the origin of the 0%/82% asymmetry.

---

# Addendum — the two results that matter most

## A. Coadding suppresses the variance leak (not just improves S/N)

Four-arm transplant protocol applied to an untrained ONIR **coadd** scan instead
of to v2, at z 1.35-2.0 — the band where v2 fails. Positive control passed
(observed arm, N epochs, median abs dz = 0.0567; gate required <0.15 on >=20 obj).

| arm | epochs | n | median abs dz | TARGET lock | VAR-SRC lock |
|---|---:|---:|---:|---:|---:|
| matched (own envelope) | N | 90 | 0.0062 | **92%** | **0%** |
| **mismatched (transplanted, z 0.4-1.2)** | N | 78 | 0.0054 | **88%** | **0%** |
| noise only | N | 78 | 0.8069 | **3%** | 3% |
| all arms | 1 | ~88 | ~0.8 | 3-14% | 0-2% |

Head to head with v2 in the same band:

| | v2 (z 1.68-2.0) | coadd scan (z 1.35-2.0) |
|---|---:|---:|
| signal + mismatched envelope -> target lock | **35%** | **88%** |
| noise alone -> target lock | **68%** | **3%** |

v2 follows the transplanted envelope more often than the signal. The coadd scan
never does — variance-source lock is 0% in every arm. Mechanism: source adds
coherently across epochs, noise incoherently, so signal-to-leak improves as
sqrt(N). Visible in the 1-epoch rows, where the advantage has not yet accrued.

**Consequence: the z~1.4 boundary is a property of v2's per-visit-logit
architecture, not a limit of the method.** This is a second, independent
justification for the coadd representation.

Caveats: untrained template matching, so it validates the REPRESENTATION only — a
trained coadd model could still learn the shortcut from labels, which needs the
four-arm test on NERSC. `observed` scored 52% vs `matched` 92%, so real correlated
noise (lag-1 ~0.9) is harder than fresh Gaussian and absolute numbers are
optimistic. n=78-90, one configuration.

## B. The timing leak is far worse than item 2 suggested

A single scalar with **zero spectral content** classifies Ia vs CCSN:

| feature | accuracy |
|---|---:|
| observer-time **span** alone | **90.5%** |
| early->late **gap** alone | **88.5%** |
| available epochs before subsampling | 75.3% |

Gap-only beats the claimed spectral EVOLUTION result (75.5%) in **every** redshift
bin. The N_FIX=8 control built to exclude this failed because it subsampled
uniform in RANK, not phase, and always spanned each object's full range — leaving
Ia at median 64.8 d and CCSN at 135.5 d.

**Therefore: a timing-only control arm is MANDATORY on every classification run,
not a post-hoc check. Acceptance criterion: timing-only must sit at chance.**
No classification number on this simulation is admissible without it.

## Next two decisive experiments

1. **Timing-only control after cadence augmentation.** Until span-alone drops from
   90.5% to chance, no classification result means anything.
2. **Four-arm transplant against a TRAINED coadd model** (NERSC). Addendum A shows
   the representation is clean; only this shows whether training reintroduces the
   shortcut from labels — and it decides whether z 1.4-2.0 is recoverable.
