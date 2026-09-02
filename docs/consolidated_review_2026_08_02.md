# STRIDER — consolidated review and implementation specification

> **Historical research record:** This page preserves an earlier experiment or
> decision. It does not define the current STRIDER tool. See
> [`architecture.md`](architecture.md) for the current design.

**Date:** 2026-08-02
**Source:** adversarial reviews of STRIDER 2 and the STRIDER design documents,
plus an independent review of the STRIDER local pilot.
**Status:** review complete. This document supersedes the individual reports.

---

## 0. How to read the evidence tags

Every claim below carries a tag. **Do not treat them as interchangeable.**

| Tag | Meaning |
|---|---|
| **VERIFIED** | A fact about code or data, checked directly with file:line or an executed measurement. Not a judgement. |
| **RESOLVED** | A controlled comparison whose effect exceeded a pre-declared margin with adequate seeds. Actionable. |
| **UNDERPOWERED** | Tested, but the data cannot distinguish the arms. **Not a null result.** Decide on simplicity/cost/correctness grounds and record why. |
| **UNRESOLVED** | Open question. A named experiment settles it. |
| **REFUTED** | Previously believed, now disproven. |
| **SYNTHETIC** | Established only in a synthetic analogue. Tests mechanism, not supernova accuracy. Never quote as an astrophysical number. |

A recurring failure in this project has been reporting UNDERPOWERED as if it were
RESOLVED. With 2 seeds the 95% CI on the seed standard deviation spans **71×**; at
5 seeds, 4.8×; at 10 seeds, 2.7×. **Minimum 5 seeds for any seed-variance claim,
10 for a headline.** Adding test objects does not reduce seed variance.

---

## 1. Executive summary

STRIDER's distinctive design is sound and should be preserved: named rest-wavelength
anchors, an explicit redshift scan, learned local profiles, joint class/redshift
inference, phase-dependent evidence, multi-epoch accumulation.

Three things are established well enough to act on:

1. **The named ONIR anchors carry real information.** Randomising anchor positions
   within the same range costs −0.205 macro-F1 (20 seeds, paired, 10× the margin).
   An earlier 2-seed study reported this as "no benefit"; that was a power failure.
   *(RESOLVED, SYNTHETIC)*

2. **The v2 atlas and bank have structural defects that must not migrate.** The 15
   windows are only ~8.6 independent measurements; 28.5% of bank cells are fabricated
   or hold a single noisy window; 63% of class pairs have identical declared feature
   sets; the vocabulary is optical while the instrument is near-infrared, leaving
   exactly one feature in band at both z=0 and z=3. *(VERIFIED)*

3. **The local pilot's redshift is dominated by observing schedule, not spectra.**
   Cadence-resistant training must land before any ONIR comparison, or every ONIR arm
   is graded against a model whose redshift is partly supplied by the schedule.
   *(VERIFIED from the pilot's own controls)*

One thing is **not** established and gates the largest decision in the programme:

> **The mechanism of the STRIDER-2 high-z variance route is unknown.** Source-Poisson
> is refuted (measured d log σ / d log F = **+0.001**, where theory predicts ~+0.015
> at the measured source fraction p≈0.03, and where a leak would need far more).
> Epoch-common noise is refuted (cross-epoch residual correlation +8e-5 over 543,414
> pairs; the null bounds any common component at ≲0.1% of residual energy). The
> leading remaining hypothesis — a rest-frame-locked beat between the simulation's
> template grid and the observer bin grid — would make the channel a **simulation
> artefact that does not exist in real Roman data**. *(UNRESOLVED)*

**The channel itself is real, survives its controls, and is redshift-specific.** A kNN
on the object-specific residual log-FLAMERR *shape* recovers z at median |Δz| **0.0027**
(98.3% within 0.05; Ia, 2.2≤z<2.7, N=1207) versus **0.0102** for the residual
clean-signal shape. Controls run: blocking on `SIM_LIBID` (0.0027) and on HEAD file
(0.0029) leave it intact; truth z never enters the predictor, only the target; the
native grid and band edges are identical across objects by construction; a
target-permutation null gives 0.114–0.126. Blocking `SIM_TEMPLATE_INDEX` is degenerate
because all SALT3 Ia share one value — that dissolves the confound rather than failing it.

The decisive part is **specificity** — the same σ-shape predicts redshift and almost
nothing else:

| target from the same σ-shape | skill vs its own null |
|---|---|
| **redshift** | **+0.978** |
| SIM_PEAKMAG_J | +0.229 |
| SIM_PEAKMAG_H | +0.148 |
| HOSTGAL_LOGMASS | **+0.008** |
| MWEBV | **+0.009** |

At fixed brightness (z residualised on peak magnitude) skill is **+0.922**; within
magnitude quartiles median |Δz| is 0.0044–0.0062 against nulls of ~0.12. This rules out
**host-galaxy light in the background** (would predict host mass) and
**distance-modulus-via-brightness** (survives at fixed magnitude). The channel tracks
(1+z) and essentially nothing else — the signature of a (1+z)-locked grid beat.
*(VERIFIED under the named controls.)*

**Consequence:** a whitened retrain is currently un-justified. If the channel is a
simulation artefact, whitening is the wrong response entirely and the high-z
programme is re-scoped rather than retrained. **Experiment X1 (§7) settles this in
about half a day with no GPU and should run before anything expensive.**

---

## 2. The design table: STRIDER 2 → STRIDER

`ADAPT` = carry the idea, change the implementation. `ADD` = new. `KEEP` = unchanged.
`DROP` = remove. `FIX` = defect repair.

### 2.1 Data and storage

| # | Component | STRIDER 2 (verified) | STRIDER | Action | Why | Evidence |
|---|---|---|---|---|---|---|
| D1 | Storage | `*.npy.gz` per field, globally padded to 64 epochs, fully decompressed into RAM. 22.8 GiB at 15 epochs, 97.5 GiB at 64. | Ragged native-bin HDF5 + offsets; Parquet as a *derived* index only. | ADAPT | Padding wastes 75–87% of compute (typical object has 8–15 of 64 epochs). Native bins are required because **noise must be generated before resampling**. | VERIFIED (arithmetic matches both figures exactly) |
| D2 | Wavelength storage | n/a | **Do not store per-bin edges.** All 167,919 epochs checked share one 201-bin ROMAN_PRISM master grid; store a master grid + `uint8` index. | ADD | 8 B/bin → 0.011 B/bin, and it stops every loader treating each epoch's grid as arbitrary when it is fixed. | VERIFIED (0 exceptions in 167,919 epochs) |
| D3 | Precision | float32 | float32. **Forbid float16 without a recorded per-array scale.** | FIX | FLAM ~1e-19 cgs: 100% of 19.2M values underflow float16 to zero. An earlier "158× compression" was compressing zeros. | VERIFIED |
| D4 | Compression | gzip | byte-shuffle + gzip4 (12.03 → 8.82 B/bin). Budget 4 B/bin/array, not 1.5. | ADAPT | Measured on real arrays; plain gzip achieves only 1.07×. | VERIFIED |
| D5 | Realistic size | — | 3 flux arrays + index + valid ≈ **8.8 GiB** at 177k objects × 30.2 epochs — ~5× smaller than the current padded 1024-bin store at 64 epochs. | — | Native grid is 201 bins, not 1024. | VERIFIED |
| D6 | Covariance operator | — | **Not needed for this simulation.** Native-bin noise is independent (lag-1 autocorr −0.005 = exactly the −1/(n−1) mean-subtraction bias). Retain the hook for the alternative-ETC arm. | DROP | Measured. All bin-bin correlation in model inputs is *created by the resampler* (0.95 at lag 1) — which is the measured justification for native-bin generation. | VERIFIED |
| D7 | Count-space Poisson | — | **Not needed.** 83–10,462 e⁻/bin, background-dominated; Gaussian error ≤11% in skewness at worst. Keep `count_to_flux` as a diagnostic. | DROP | Removes a whole subsystem from the critical path. | VERIFIED |
| D8 | Three flux arrays | — | Keep `FLAM` (only training input), `SIM_FLAM` and `FLAMERR` — but put the latter two behind a `simulation_only/` boundary enforced like `allowed_use: evaluation`. | ADAPT | `SIM_FLAM` is a generator input and diagnostic; `FLAMERR` is the array under indictment. Same group + same access pattern is how a truth channel is born. | VERIFIED (no leak today: one read site, `analysis/diagnostics/hg2_snana_loader.py:104`) |
| D9 | RNG / resume | No resume exists anywhere in `training/*.py`. | Counter-based key `(source_id, epoch_id, training_epoch, realization, noise_model_id)` via `blake2b`. | ADD | Proven exact: identical draws at 0/1/2/4/8 workers, and across a 4→2 worker change on resume. A naive per-worker `torch.Generator` **replays** consumed noise on restart. | VERIFIED (executed) |
| D10 | DataLoader workers | Full in-RAM arrays; no HDF5. | Lazy per-worker handles guarded on `os.getpid()`; **sharded** store; `persistent_workers=True`. | ADD | The pilot measured 548 → 3.4 obj/s from 0 → 12 workers. Cause is *not* inherited handles (the pilot already opens lazily at `dataset.py:289` and clears in `__getstate__`) — repeated worker startup, per-worker caches and single-file contention are the live suspects. `num_workers: 0` is a Mac-specific workaround and **must not** go to NERSC. | VERIFIED (code + pilot benchmark) |

### 2.2 Noise, uncertainty and the leak

| # | Component | STRIDER 2 | STRIDER | Action | Why | Evidence |
|---|---|---|---|---|---|---|
| N1 | Training realisation | One fixed SNANA `FLAM` realisation | Fresh native-bin realisation per read, from the D9 key | ADAPT | Required for the noise-robustness programme. **Note this is also unbounded augmentation** — it changes effective dataset size and optimal regularisation independently of any whitening, and must be isolated as its own ladder rung or it will be credited to the wrong change. | Design |
| N2 | Input scaling | Per-object, jointly across epochs, of raw `FLAM` | Divide valid bins by a **source-free** scale, then centre/scale | ADAPT | Removes the source-dependent route from the input. Note: this is justified on **signal-quality** grounds — a direct test found σ_total normalisation did *not* create a leak (decoy slope ~0), it only degraded clean performance. | REFUTED (that it is the leak route); design stands |
| N3 | Source-free scale | — | 1/A(λ), background+read only. **Constructible locally today** as the median variance over the faintest decile per (bin, exposure condition); A is stable to ≤3%. | ADD | Removes a blocking dependency on new simulations. | VERIFIED |
| N4 | Epoch weights | `combiner_mode='snr_sum'`, weights ∝ 1/σ²_total, `ivar_weight` emitted unconditionally | Uniform or exposure weights first; 1/A later as an isolated arm. **Never 1/σ²_total.** | FIX | Two independent problems. (a) **Safety:** 1/(A+B·f) is strictly monotone in the redshifted source — measured rank correlation with clean flux: uniform 0.000, exposure 0.000, 1/A 0.000, **1/FLAMERR² −0.318**. (b) **Correctness:** per-epoch evidence is an NCC whose noise variance is σ-independent, so the optimal weight is ∝1/σ, not ∝1/σ². | VERIFIED (a); analytic (b) |
| N5 | Coadd weighting | n/a | **Prohibition is load-bearing, not cautious.** Extend it to the depth/exposure metadata channel — anything derived from Σ1/σ²_total carries z. | ADD | The `co_ivar` arm produced the best-looking model in the whole review (median\|Δz\| 0.0019, 7.6× better; accuracy 0.996; coverage-68 0.982) and was a pure noise reader: **collapses on clean input (0.352), recovers z from source-free noise (lock 0.970), follows a foreign envelope 98.1% of the time, covers true z 3.5% of the time.** | RESOLVED, SYNTHETIC |
| N6 | Clean-source control | Absent | **Mandatory and blocking on every arm.** | ADD | It is the *only* internal metric on which `co_ivar` looks bad. Rule: any arm that improves noisy-data accuracy while degrading clean-source accuracy has learned the noise. | RESOLVED, SYNTHETIC |
| N7 | The leak mechanism | Assumed source-Poisson | **Unknown.** Do not build the fix on the assumed mechanism. | UNRESOLVED | Source-Poisson refuted (slope +0.001 vs ~+0.015 predicted). Epoch-common refuted (+8e-5, bounds ≲0.1%). Host light and distance-modulus refuted by the specificity profile in §1. Leading hypothesis is a rest-frame-locked template/bin-grid beat = simulation artefact. | REFUTED ×4; **X1 settles** |
| N7a | **Transplant is S/N-matched** | — | Record as the evidence line for the 70% donor lock. | — | On the exact 100 target/donor pairs the FLAMERR amplitude ratio is **median 1.004** (IQR 0.96–1.05) and per-bin clean-signal S/N is **0.128 matched vs 0.133 mismatched**. This is what makes the lock a **necessity** result rather than "the signal was drowned". Without it, risk 1 is arguable. | VERIFIED |
| N7b | **Whitening may be self-defeating** | Plan assumes whitening closes the leak | **Do not assume whitening is the fix, even if X1 returns "real instrumental effect".** | ADD | The standardised residual (FLAM−SIM_FLAM)/FLAMERR has **sd 1.0005, mean 0.0000** — so the leak lives in the σ **field**, not the realisation. Pointwise whitening by FLAMERR divides the *signal* by the same z-bearing shape, imprinting it on the signal path. This predicts the recorded `observed_whitened = 0.644`. **X1 must therefore also test whether *any* pointwise rescaling can separate the channels**, not only where the pattern is locked. | VERIFIED |
| N8 | CCSN template cutoff | Unknown | **Mask or flag.** NON1ASED red limit at rest **10331.3 Å** (NMAD 0.21%) produces `FLAMERR` spikes >1000× the epoch median in ~20–33% of CCSN epochs. Feature-free z readout at NMAD 0.0029. 100% at z<0.5, 0% at z>1, 0.1% in SALT3 Ia. Survives resampling; no mask flags it; `|FLAM|` there is 4.4e5× the good-bin median, so per-object normalisation is set entirely by those bins. | ADD | Every CCSN result below z=1 is confounded until fixed. This is **separate** from the variance route (which is broadband, not an edge). | VERIFIED (independently confirmed) |

### 2.3 Encoder

| # | Component | STRIDER 2 | STRIDER | Action | Why | Evidence |
|---|---|---|---|---|---|---|
| E1 | Stem | Single-scale CNN, RF **22 bins = 141–339 Å** (from `strider/layers.py:263-271`: k=[7,5,5] stride-1 then k=8 stride-8; earlier drafts said 18 bins/115–277 Å, which is not derivable) | **Dilated residual multi-scale**, RF 68 bins = **436–1046 Å**. Single trunk, residual accumulation over dilations 1/2/4/8, **no branch-concat, no BatchNorm**. | ADAPT | +0.026 F1 and −0.0021 median\|Δz\|, tightest seed variance in the study, +1.4% params. Physical, stated in **bins** so it is z-invariant: an 18,000 km/s trough spans **70 bins** at every z, against the current stem's **22** and the dilated stem's **68**. (Quoting Å widths invites comparing an observed-frame width at one λ against an RF range spanning the whole band.) **The current stem cannot see a whole trough in one token.** Gain concentrates in truncated coverage and long phase span. | RESOLVED, SYNTHETIC |
| E2 | The prior rejection | "Multi-scale was tested and rejected (lost on OU)" | **Restate as: "branch-concat + BatchNorm multi-scale was rejected."** | FIX | That implementation confounded three things: parallel branch-concat, BatchNorm (which the audit notes exists *only* in the multi-scale stem and interacts with epoch-count batching), and the receptive field. | VERIFIED (code) |
| E3 | OU gate | — | **The dilated stem must pass OpenUniverse before adoption.** | ADD | Every encoder number here is in-distribution on one synthetic generator — exactly the condition under which the rejected stem also looked fine. | UNRESOLVED |
| E4 | Attention organisation | Factorized wavelength-then-time | **Keep factorized.** | KEEP | Joint attention is better (+0.036 F1) but costs **3.5× peak memory and 5× encoder time**; at production scale (64 epochs × 128 patches = 8192 tokens) the quadratic term is prohibitive. ViViT's own result is that the factorised encoder wins under data scarcity. | RESOLVED (both the gain and the cost) |
| E5 | Local windowed attention | — | **Reject.** | DROP | −0.082 F1, RESOLVED-WORSE. Windows sever the long-range line pairings that break redshift aliases (Ca II H&K ↔ Ca II NIR is 0.34 dex apart). | RESOLVED, SYNTHETIC |
| E6 | Position encoding | Absolute, `use_spectral_alibi: false` | **Keep. Question is settled.** | KEEP | Absolute-only vs ALiBi-only are indistinguishable (**314 seeds/arm** to separate). Removing position entirely is catastrophic (F1 0.39, median\|Δz\| 0.186). The docs' concern that absolute encoding is "an unaudited bypass route" is not supported — it is redundant with ALiBi, not a bypass. | UNDERPOWERED between schemes; RESOLVED that one is required |
| E7 | Front-end nonlinearity | Not considered a design variable | **Treat as leak-relevant.** | ADD | On a source-deleted probe the GELU conv stem reached **96%** of the analytic oracle; linear patch projection reached **37%**. A rectifying front end converts local noise amplitude into a positive feature; a zero-mean NCC match cannot. Independently confirmed in two analogues. | RESOLVED, SYNTHETIC |
| E8 | Channels | Transformed flux + temporal difference (`n_channels=2`) | **Keep the two channels.** | KEEP | Validity channel (+0.0095 F1, needs 46 seeds), coadd channel (+0.0149, needs 15), both (+0.0097) — **all UNDERPOWERED.** Recommend two channels on **parsimony**, not on the metric. | UNDERPOWERED |
| E9 | Scan cost | — | Encoder runs **once**; the scan gathers from its tokens. | KEEP | Marginal scan cost is 0.7–3.1 ms against a 17–151 ms encoder. Any candidate requiring a re-run per trial redshift multiplies cost by Z (=500) and is disqualified on that ground alone. | VERIFIED |

### 2.4 ONIR atlas and signature bank

> ⚠️ **Power caveat for this whole section.** The ONIR analogue's own positive control
> **failed**: dropping a class-critical feature gave f1 Δ = −0.0083 [−0.0255, +0.0089]
> against a 0.020 margin — **UNDERPOWERED, n_req=13**. Therefore **no claim about the
> effect of an individual feature is admissible at 20 seeds.** This does *not* touch
> A1, A9 or A11, whose effects are 10–22× the margin, nor any VERIFIED code/bank fact.
> It does mean the 23-feature proposal (A3) is a **coverage argument, not a measured
> per-feature gain.**

| # | Component | STRIDER 2 (verified) | STRIDER | Action | Why | Evidence |
|---|---|---|---|---|---|---|
| A1 | Anchors | 15 named rest wavelengths | **Keep the design.** | KEEP | Randomising positions within range: **−0.205 F1**; unconstrained: **−0.450**. Both RESOLVED-WORSE at 20 seeds, paired. The exact wavelengths matter *and* a rest-frame coordinate system matters. ⚠️ The random-position arm also has **3.7× the seed SD of baseline** (0.1363 vs 0.0369) — it is *unstable*, not merely worse. Change 2's ladder uses random-position anchors as a control and will inherit that variance; budget seeds accordingly. | RESOLVED, SYNTHETIC |
| A2 | Effective independence | 15 windows, mean adjacent overlap 46% (worst pair 85.7%) | Report the **effective** count. | FIX | Union of all 15 windows = **8.57 window-widths** (at the correct 120-bin window; 9.02 if the erroneous 113-bin figure is used). z-invariant, since redshift is a rigid translation in log λ. They are ~8.6 independent measurements, not 15. | VERIFIED |
| A3 | Wavelength coverage | Optical vocabulary (3945–8579 Å) on a NIR instrument | **Add 8 features**: rest-NIR (`FeII_CoII_10500`, `CI_MgII_10800`, `HeI_10830`, `Paschen_beta_12818`, `Ia_Hband_break_15400`) and rest-UV/blue (`MgII_FeII_2800_UV`, `SiII_3858`, `OII_W_4100`). 15 → 23. | ADD | **Criterion: FULL-window support** (the whole ±7-patch window in band). On that criterion the count is **1** at z=0, **1** at z=3, peak 14 at z=1. *(On the weaker ANY-overlap criterion it is 2 at z=0 — state which criterion you mean; the two appear elsewhere in this review and are not interchangeable.)* Proposed: z=0→6, z=1→15, z=2→12, z=3→5. This also *causes* the bank pathology (A13). **Coverage argument, not a measured per-feature gain** — see the §2.4 power caveat. | VERIFIED |
| A4 | **Coverage/aggregation coupling** | `mean` over fixed F=15 | **Fix aggregation before extending coverage.** | FIX | **Mechanism (VERIFIED):** zero-support features contribute exactly 0.0 under `mean`-over-fixed-F, which is the *indifference* value; adding features that are out of band at some z therefore adds indifference mass exactly where coverage is worst. See A5 for the 150× high-z limit. **The order is retained because it is mechanistically motivated and costs nothing — not because the effect size is established.** ⚠️ The previously quoted effect sizes (+20.1% / +0.044 / +46.6%) are **untraceable in any captured artifact** and one of them collides with an unrelated figure (`audit_out.txt:198`, "20.1% of populated cells outside declared phase support"). Do not quote them. | **Mechanism VERIFIED; effect sizes UNVERIFIED** |
| A5 | Aggregation rule | `mean` over fixed F; zero-support features contribute exactly 0.0 | Coverage-weighted mean + additive log-coverage term: `Σ_f cov_f·m_f / Σ_f cov_f + (β/T)·log(Σ_f cov_f / F)`, β=1, with **fractional** coverage. | ADAPT | 0.0 is identical to "indifferent" and strictly better than any anti-correlated match. Per-feature the effect is diluted by 1/F (~1.4×), but in the all-unsupported high-z limit it is exp(0) vs exp(−5) ≈ **150×**, and `snr_sum` **sums** over epochs so the gap scales with epoch count. A structural high-z attractor independent of the variance leak. | VERIFIED (mechanism); RESOLVED (fix, SYNTHETIC) |
| A6 | `overlap_norm` | Implemented, tried, "blew up mid-z aliasing" | **Do not adopt as specified in the audit.** Use the log-coverage term instead. | FIX | **Root cause (VERIFIED):** `edna_detector.py:262` uses `gather_valid.any(dim=-1)`, so a feature clinging to the band by **1 patch of 15 gets full weight**. That is a sharper explanation than "it deleted the coverage prior", and the recommendation stands on this alone. The analogue's repeat of the failure is **UNDERPOWERED** (f1 Δ=−0.041 [−0.063,−0.019] against a 0.020 margin, n_req=22) — directionally consistent, not resolved. | **VERIFIED (root cause); UNDERPOWERED (repeat test)** |
| A7 | Window geometry | One global `window_radius_patches=7` = ±15,389 km/s for every feature | **Per-feature, asymmetric, `window_mode`-keyed.** Absorption gets blueward-only; p_cygni asymmetric; region/blend two-sided. | ADAPT | **Adopt on cost, not accuracy** — it is EQUIVALENT to simply widening to r=12 on F1, but reaches it for +16.5% window bins vs r=12's +66.7%, and on the existing 15 features it is **−28% compute** and drops >25%-overlap pairs from 16 to 9. Report it as a cost argument. | UNDERPOWERED (accuracy); VERIFIED (cost) |
| A8 | Dead metadata | `velocity_range_kms`, `phase_range_days`, `window_mode` parsed at `strider/features.py:107-108` and **read nowhere** in the detector. The atlas NPZ even ships a precomputed `window_half_bins_by_z_feature (19,15)` that nothing reads. | **Wire all of them.** | FIX | The detector's only physical input from the atlas today is `feature_rest_waves` — 15 scalars. The per-feature geometry is declared-but-dead *twice over*. | VERIFIED |
| A9 | `target_classes` | Declared but unused | **Soft prior on aggregation weight. NEVER a hard mask.** | ADAPT | Hard masking was the **single worst arm in the entire review** (F1 −0.314). A feature's *absence* in a class is evidence; masking deletes it. | RESOLVED, SYNTHETIC |
| A10 | Class discrimination | 63% of class pairs (66/105) have **identical** feature sets; only **4** distinct sets for 15 classes under the expansion that yields the 66/105 figure (6 unexpanded; earlier drafts said 7, which is consistent with neither); SLSN has no unique marker | Give classes genuinely distinct features (O II W-feature for SLSN, He I 10830 for He-rich, Pa β for H-rich). | ADD | Twelve of fifteen classes are declared to be diagnosed by the same ten `non-Ia` features. | VERIFIED |
| A11 | Bank prototypes | Medoids of individual noisy training `FLAM` windows | **Clean class-mean templates.** | ADAPT | +0.064 F1 and **+0.164 rare-class recall** vs medoids. Random-window prototypes are worse still (+16.0% median\|Δz\|). | RESOLVED, SYNTHETIC |
| A12 | Empty cells | `phase_neighbor_fallback` fabricates them, copies the class's own signature, sets `active=True`, picks nearest by **bin index** on an uneven grid (so "nearest" can be 20 d away) | **`min_train_windows: 25`; back off to an `evidence_group`-shared prototype, then mask. Never fabricate.** | FIX | `support_flag` = {0: **482** (13.4%), 1: **543** (15.1%), 2: **2575** (71.5%)} of 3600. ⚠️ **The earlier "28.5% fabricated or single noisy window" mislabels these.** Verified semantics: flag=0 = **under-populated at build** (0–19 windows, median 5) — these are what `phase_neighbor_fallback` later fabricates; flag=1 = a **mean over 20–99 windows** (median 45), the *cleanest* prototype class; flag=2 = **single-window medoids** selected from 100–190,691 candidates. So the correct statement is: **13.4% under-populated, 15.1% low-N means, and 71.5% single-window medoids** — which makes A11 (clean templates) *more* important, not less. KN is severely under-populated — **10 of 16 phase bins have zero supported cells**. ⚠️ The per-class fabricated-cell *count* is disputed across three computations (157 / 87 / 199 of 240) and depends on whether the bank's class axis matches `metadata['class_names']`; **pin it before quoting.** The qualitative finding holds under all three. Fabricated cells can only *inflate* that class's score. | VERIFIED |
| A13 | Dead-cell cause | Assumed sample size | **Atlas geometry.** KN's three permanently-dead features are exactly the three with the highest in-band z thresholds, perfectly separated by that threshold. | — | Coverage determines which cells can *ever* be populated. Fixing A3 fixes part of A12. | VERIFIED |
| A14 | `signature_active_fraction` | Computed **after** fallback; pinned near 1.0 by construction (the code says so at `edna_detector.py:438-444`) | **Report pre-fallback real support.** | FIX | The system's own health metric cannot report the failure it exists to catch. ⚠️ Correction: the metric reads **0.688–0.798** (mean 0.718), **not** "near 1.0" — its denominator is all K=8 prototype slots, not cells (at cell level it is 0.964). The mechanism claim stands; the number in earlier drafts did not. | VERIFIED (mechanism); number corrected |
| A15 | NCC normalisation | Two conventions coexist; the deployed path is the **asymmetric** one | **One symmetric NCC.** | FIX | A *perfect* match over a partly-in-band window scores √(in-band template power) — 0.707 at 50%, 0.316 at 10% — because `g` is masked at `matcher.py:32` but `t_norm` sums the whole window at `edna_detector.py:602`. Class-dependent bias, worst exactly where features leave the band. Adopt on **correctness**; the accuracy difference is UNDERPOWERED (3 seeds). | VERIFIED |
| A16 | `k_max` / `top_k` | k_max=8, top_k=3 | **k_max=2, top_k=1** | ADAPT | Both UNDERPOWERED (7–13 seeds needed). Adopt on cost and defensibility: "best-matching prototype" is a claim you can defend; "mean of the best three of eight" is not. 543 real cells already have k=1 regardless. | UNDERPOWERED |
| A17 | New schema fields | — | `evidence_group`, `window_offset_kms`, `window_half_patches`, `z_full_support`, `min_train_windows` | ADD | `evidence_group` stops the aggregator double-counting one physical complex (the 1 µm region is one complex, not three lines). | Design |
| A18 | Alias structure | — | 146 (z, pair) combinations are structurally unbroken, all at band edges. | — | Real but below the threshold to drive design at 1.3% ridge prevalence. Removing the alias breaker is EQUIVALENT (CI excludes zero but lies inside the margin). | UNDERPOWERED |

### 2.5 Phase, time and cadence

| # | Component | STRIDER 2 | STRIDER | Action | Why | Evidence |
|---|---|---|---|---|---|---|
| P1 | Phase to encoder | **Present.** `phase_film` in the frozen weights, built from truth z and truth peak time | **None.** Remove the parameter from the signature — do not make it `None`-able. | FIX | The encoder is conditioned on 1/(1+z_true) — the quantity being inferred. | VERIFIED |
| P2 | Phase in scan | `φ = Δt_obs/(1+z)` exists at `edna_detector.py:741`, but the scan has **no t₀ axis** | Full `(z, t₀)` scan; marginalise over a measured p(t₀). | ADAPT | Every "candidate phase" result to date is therefore the **true-peak diagnostic**, not the deployable setting, and should be relabelled. | VERIFIED |
| P3 | Truth seam | `ObserverPhases = MJD − SIM_PEAKMJD` (`convert_snana_fits.py:629`); raw MJD read at `:606` then **discarded** | Retain float64 MJD; represent t₀ separately as a distribution. | FIX | z-free but **not t₀-free**. This is a **truth seam**, not a realism gap: every pilot redshift number is an **upper bound**. | VERIFIED |
| P4 | Peak-time modes | Truth only | `truth_control` (diagnostic, **not exportable**), `point_peak`, `marginalize_peak` (primary), `no_peak` | ADD | Integrating over a measured p(t₀) recovered the true-t₀ upper bound (coverage 0.93) where a **point** estimate silently lost coverage (0.87). Go to the distribution, not to a fitted point. | RESOLVED, SYNTHETIC |
| P5 | Phase support | Silent clamp at `matcher.py:146-149`; no flag, no attenuation | Explicit support weight, 0 outside; wire per-feature `phase_range_days`. | FIX | **Clamp exists and is silent (VERIFIED).** **684 populated cells (20.1%) sit outside their own declared phase support** and are scored as measured evidence (VERIFIED). Clamping is systematically low-z-loaded, because φ=Δt/(1+z) compresses toward the supported region as z rises — a structural high-z attractor from grid geometry alone. ⚠️ The "**18.9%** of the (Δt,z) plane" figure is **UNVERIFIED** — it may describe a rectangle the analysis drew rather than the empirical Sundial (Δt,z) distribution. Use the sourced 20.1% instead. The fix **in isolation is UNDERPOWERED** (n_req=4); it is RESOLVED only **jointly with A5** (support + log-coverage, +0.066 f1). | **VERIFIED (mechanism); UNDERPOWERED (fix alone); RESOLVED jointly with A5** |
| P6 | **Cadence robustness** | n/a | **Paired source/no-source training with identical visit times**, + z-independent visit **span** sampling, + timing-only baseline. | ADD | **This is the first thing to build.** See §3. | VERIFIED (pilot controls) |
| P7 | Time-dilation channel | — | Retain as *alias-veto*, not precision-z. | ADAPT | Timing-only recovers the shift at **exactly chance** while recovering class at 2.7× chance. With t₀ free, N epochs give the temporal channel **N−2 dof**; for N≤2 it is *exactly* uninformative about z. The intrinsic-stretch floor (σ_z ≈ 0.15–0.20 at z=0.5–1) is **larger** than the Si II↔Hα alias separation it was meant to veto. | RESOLVED, SYNTHETIC |
| P8 | Conflict behaviour | — | Disagreement must **broaden or split** the posterior. | ADD | Metric: `CBR = H_conflict/H_consistent`, violation = CBR < 1.5 **and** coverage of the true value < 0.5× nominal. A learned fusion gate **narrowed** under conflict (z68 0.021 vs 0.325) — reject that form; a fixed scalar correctly broadened (0.215 vs 0.122). | RESOLVED, SYNTHETIC |

### 2.6 Epoch combination and coaddition

> WARNING - **Seed-count caveat for this whole section.** Every coadd comparison below is
> **n = 2 seeds with no confidence intervals** - which by this document's own Rule 1
> cannot support a seed-variance claim (the 95% CI on the seed SD spans 71x at n=2).
> The `co_ivar` catastrophe (N5, C4) survives regardless because its effects are 22-66x
> the margin across four independent diagnostics. **Everything else in 2.6 should be
> read as directional.** The four `co_ivar` numbers also come from four different eval
> arms (`arm1_orig`, `pc_clean`, `arm4c`, `arm3_mismatch_env`) - each correct in its own
> arm - and the 98.1% donor-following is **not unique to `co_ivar`**: `coadd_only_ivar`
> gives 0.981 too.

| # | Component | STRIDER 2 | STRIDER | Action | Why | Evidence |
|---|---|---|---|---|---|---|
| C1 | Combiner | `snr_sum`, ∝1/σ²_total, unknown modes **fall through silently** to coverage weights (`combiner.py:96-104`) | Validated enum (raise on unknown); weight ∝1/σ; rename — it is neither S/N nor a variance | FIX | Silent fallthrough is a live foot-gun. | VERIFIED |
| C2 | Aggregation and confidence | `sum` over epochs at fixed temperature | Make the scan temperature **evidence-dependent**; calibrate coverage **stratified by epoch count** | ADAPT | On **pure noise**, entropy fell 1.92→1.09 and max class probability rose 0.45→0.70 as epochs went 1→8, at chance accuracy — confidence manufactured by epoch count, because `TEMP × Σ_T` makes effective temperature scale with T. Neither `sum` nor `mean` yields a width that tracks evidence. **This reopens the recorded "kept `mean`" decision.** | RESOLVED, SYNTHETIC |
| C3 | Coadd — replacement | — | **Reject.** | DROP | Three independent studies agree: 8–19% worse (analytic Monte Carlo, 20k realisations), ~2× worse (sonar analogue), **+55% worse at low S/N** (barcode analogue). For a linear-Gaussian likelihood with a time-constant source the coadd is a *sufficient statistic* and adds exactly zero. | RESOLVED |
| C4 | Coadd — fused branch | — | **Adopt-later**, `fuse_fix` (fixed scalar) form only, low-S/N only. | ADAPT | −20 to −30% median\|Δz\| at low S/N with ≥2 epochs; **~0% at high S/N; ~0% on classification** (the latter two UNDERPOWERED, not null). **The gain is bought with coverage**: the two biggest point-estimate gains drop coverage-68 to 0.53 and 0.56 against a 0.693 baseline. Only `fuse_fix` (−20%, coverage 0.656) is a fair trade. | RESOLVED (low-S/N gain); UNDERPOWERED (elsewhere) |
| C5 | Coadd — post-hoc fusion | — | **Reject.** | DROP | Best point accuracy in the study (0.0124) with coverage-68 **0.622** and no-source lock 0.023→0.097 (**4×**). Double counting, measured. | RESOLVED, SYNTHETIC |
| C6 | Coadd — encoder | — | **Shared encoder + adapter.** Not a dedicated encoder. | ADAPT | Matches a dedicated encoder at 27% fewer parameters and higher throughput. The ONIR scan must stay visible in the coadd path. | RESOLVED, SYNTHETIC |
| C7 | Coadd — form | Plan: "object-level branch, not a per-epoch channel" | **Demote to an open question.** | FIX | Both were built. The *rejected* channel-copy form scored ΔF1 **+0.0149**; the *preferred* pseudo-epoch form scored **−0.0037**, and was the only arm with an elevated variance-envelope slope and the highest blank-input confidence (0.943). Both UNDERPOWERED — but the plan states the preference as established when the evidence weakly points the other way. | UNDERPOWERED |
| C8 | Coadd — span | Plan: time-grouped / phase-aware coadds | **One fixed ~20 observer-day span**, peak-centred. | ADAPT | **Geometric argument (sound):** rest-frame span compresses as 1/(1+z) while the safe observer span grows as (1+z), so **a single fixed span sized at the lowest redshift is automatically safe at all higher z** — which removes the "coadd inside the trial scan" variant that cannot be built without truth z. **Evidence line corrected:** do *not* cite the flat `span_narrow/mid/wide` result or the time-grouping triple — span at a **fixed evolution rate** tests sampling extent, not blur. The right knob is `vel_scale`: base 0.0120→0.0139 (**+16%**), coadd_only 0.0117→0.0157 (**+34%**) from vel 0→1.5. **Coadd degrades ~2× faster than per-epoch under phase evolution.** The toy's default `vel_scale=0.35` ≈ 2,520 km/s swing against ~4,000–6,000 real for Si II 6355, so it **understates the risk by ~2×**. **The 20 d is derived, not measured** — sweep it on real data. | UNRESOLVED (number); geometric argument sound |
| C9 | Coadd — √N gain | — | Do **not** budget for common-mode saturation. | DROP | Cross-epoch residual correlation **+8e-5** over 543,414 pairs; √N intact to N=32. A synthetic arm at f=0.4 runs at **5,000×** the measured value — sensitivity study, not evidence. Within-epoch resampling correlation does **not** spoil the cross-epoch gain. | VERIFIED |

### 2.7 Outputs, validity, calibration, metrics

| # | Component | STRIDER 2 | STRIDER | Action | Why | Evidence |
|---|---|---|---|---|---|---|
| V1 | Validity | None. `P(class)` forced to compare modeled classes | **Three distinct outputs**: conditional class/redshift posterior, adequacy, redshift quality (width/multimodality/coverage) | ADD | **Every architecture tested is confidently wrong on class given pure noise** (max P 0.42–0.94) — this is **architecture-invariant; no encoder choice fixes it.** The pilot's adequacy branch already achieves 0/500 source-free inputs above 0.5. Best thing in the pilot. | RESOLVED + VERIFIED |
| V2 | Validity gate | — | **External**, using what the shape branch discards: absolute/coadded S/N, coverage, effective epoch count, phase span. | ADD | An absolute-magnitude gate reached AUC **0.889** where posterior entropy reached only 0.743. The quantity the architecture discards by design is exactly the one that detects a target-free input. | RESOLVED, SYNTHETIC |
| V3 | Redshift quality | — | **Derive from the posterior** (width, multimodality), not a separate head. | ADD | A separate head can disagree with the posterior it describes. | Design |
| M1 | Δz convention | `(z_pred − z_true)/(1+z_true)` everywhere in code and every published number | **Keep (1+z)-normalised.** The plans' mandated **raw** Δz is a regression. | FIX | A raw 0.05 threshold is **3.64× stricter at z=3 than at z=0.1** while the plans simultaneously demand results binned by redshift. Keep raw **only** for the variance-source lock statistic, where donor separation is constructed in raw Δz. | VERIFIED |
| M2 | Scatter | Plans mandate "ordinary population scatter" | **NMAD** = 1.4826·median\|dz − median(dz)\|, always **paired with outlier fraction** | FIX | Under the audit's own data, "scatter" carries **0.055%** of its variance from the core precision; std needs **4.5×** more objects than NMAD for equal precision. | VERIFIED |
| M3 | NMAD caveat | Not stated anywhere | **NMAD is inflated by the outlier fraction**: NMAD/σ_core = 1.149 at f=0.11 vs 1.294 at f=0.19 — a **12.6% shift with an identical core.** | ADD | So "NMAD 0.0043 vs 0.0050" (16%) may be an outlier-rate change, not a precision change. Never quote NMAD without f. | VERIFIED (derived + MC) |
| M4 | Two NMAD estimators | Training uses MAD about **zero**; evaluation uses MAD about the **median** | Pick the median-centred form; delete the other. | FIX | They diverge under a biased posterior. | VERIFIED |
| M5 | Posterior score | ECE only | **CRPS primary** for redshift, **log score** for class, ECE secondary with a calibrated-null floor | ADD | No STRIDER document specifies a proper score for the redshift posterior — a regression, since CRPS is already implemented at `analysis_utils.py:241-249`. ECE is biased: a perfectly calibrated model scores >0. | VERIFIED |
| M6 | Quantiles | Hazen correct in production (`classify.py:417-437`) but violated in ~12 diagnostics | One implementation. | FIX | `posterior_windows.py:127` uses the posterior **mean**, contradicting the recorded median decision (outliers 18.7→15.2%). Two diagnostics have docstrings claiming "Hazen-style" over plain `cumsum`. | VERIFIED |
| M7 | Intervals | Central | **HPD**, split-conformal | ADAPT | The HPD 68% set averages 3.05 components and **43% of posteriors are multimodal** — central intervals are the wrong object. | Prior work |
| K1 | Calibration split | None exists (`SPLIT_PREFIX_NAMES = ('train','val','test')`) | **Five-way: TRAIN / SELECT / CONFIRM / CAL / TEST**, plus two orthogonal held-out axes (noise construction, external domain) | ADD | Temperature is currently fitted on the **test** npz and ECE reported on the same rows (`calibration_15class.py:62,73-75`); 9+ scripts fit on test halves; the calibration **method** was selected on test. CAL needs ≥~2,400 objects with ≥1,000 gold-Ia. | VERIFIED |
| K2 | Split enforcement | Split identity lives in the **filename only** | Split role travels **inside** every predictions file; append-only artefact manifest with transitive closure; role-rank rule | ADD | This is the structural enabler of every leak above. A portable model whose calibration was fitted on TEST is TEST-tainted even if its own read set is clean. | VERIFIED |

### 2.8 Software, configuration and repeatable runs

| # | Component | STRIDER 2 | STRIDER | Action | Why | Evidence |
|---|---|---|---|---|---|---|
| S1 | Padded-epoch NaN | All-masked rows → NaN in **CPU eval only**; train mode and MPS are finite. NaN is swallowed at `edna_detector.py:620-621`, producing an **exactly uniform** posterior (1/15, 1/500) | Exclude padded rows before attention; clear with `where`, not multiplication | FIX | **The audit's prescribed "assert finite" test passes on the live bug.** The correct assertion is **anti-uniformity**. Training was never corrupted (train mode is finite); CPU inference is the exposed path — the SMDC deployment is protected only by `timeseries.py:520` padding with `ones` where the training loader pads with `zeros`. | VERIFIED (executed across CPU/MPS × train/eval) |
| S2 | Config | Two configs; `configs/strider_15class.yaml` declares `z_scan_n_bins: 128`, `window_radius: 3`, `z_scan_dim: 4` — checkpoint is **500 / 7 / 8**. The `feature_head` block is never read by the trainer but **is validated** at `config.py:324-335` | One typed config; strict parsing (`raise`, not `continue`); **`test_no_dead_config_fields`** | FIX | Two of the most load-bearing numbers in the paper are wrong by ~4× in the file an author would read. An access-tracking test would have caught all 19 dead keys in one run. | VERIFIED |
| S3 | Provenance | `training_config_hash = ''`, `training_commit = '36717bb-dirty'` | Embed the **resolved config text** (not a hash) + full environment record | FIX | A hash tells you two runs differ, not how. `dirty` means the exact source may be unrecoverable. | VERIFIED |
| S4 | Resume | **Does not exist.** `grep resume training/*.py` → nothing. The published model came from a chain of `--init-from-checkpoint` + `--override-epochs` | Exact resume: model, EMA, optimizer, scheduler, scaler, RNG, sampler, step. Gate: **bit-identical** state dict *and* next batch | ADD | The published model's effective LR schedule is not reconstructible from any config — a paper-methods problem, not just engineering. | VERIFIED |
| S5 | Package identity | Two editable installs both claim `strider`; one points at a **deleted** directory. From outside the repo `import strider` resolves to the *public* package | `src/` layout; one install; import-origin check in CI | FIX | Any diagnostic run from a different cwd has been running different code. | VERIFIED |
| S6 | Scientific CLI overrides | `--override-window-radius` etc. mutate the feature gather width | **Forbid.** If it changes the numbers it lives in the config. | FIX | Such an override appears in a Slurm log and nowhere else. | VERIFIED |
| S7 | Slurm | `retrain_15class_regular_chain.sh` books **4 GPUs with no DDP** (~42 idle GPU-h/job); `retrain_15class_debug.sh` uses **regular** QOS | 1 GPU unless DDP; `smoke.sbatch` hard-coded to debug QOS with a `smoke_passed` gate | FIX | The house rule (15-epoch smoke on debug before regular) is implemented by the two scripts that violate it. | VERIFIED |
| S8 | Repo scope | Core 15,252 LOC; `scripts/`+`analysis/` 78,956 LOC; `detector.py` differs by **478 lines** between research and shipped code; `strider-public/` = 434 files, 0 tracked | Extract the verified core; delete the dead trees | ADAPT | A rewrite does not touch the 84% of the sprawl that is one-off diagnostics. | VERIFIED |

---

## 3. Implementation sequence

### Change 0 — land the metric contract, alone, first

Rule 8 says the metric change is free and should land alone. It therefore needs its own
step, or the rule is stated and then violated by omission.

**Build:** M1 (keep (1+z)-normalised Δz), M2 (NMAD paired with outlier fraction), M3
(state the NMAD-inflation caveat wherever NMAD is reported), M4 (one NMAD estimator —
median-centred — delete the MAD-about-zero variant), M5 (CRPS primary for redshift, log
score for class), M6 (one Hazen implementation; fix the posterior **mean** at
`posterior_windows.py:127`), M7 (HPD intervals).

**Why alone:** every subsequent comparison is scored with these. If they change at the
same time as anything scientific, no later number is comparable to any earlier one.

**Stop when:** every metric has exactly one implementation and the evaluation suite
reports mean ± SD over seeds, paired ΔCI, margin, and verdict.

### Change 1 — cadence-resistant training (**the first scientific change**)

**Why first:** in the pilot, no-source inputs land within Δz=0.1 **33.4%** of the time with
normal visit times and **6.2%** with times zeroed. Observer-time span correlates with
redshift at Spearman r=0.299 (p=8.6e-12) and visit count at r=0.223. Any ONIR
comparison run before this is graded against a model whose redshift is partly supplied
by the observing schedule.

**Land in two measured sub-steps — they are sequential interventions, not one change.**
By the limitation note below, the twin loss *cannot* close the gated-timing path; span
decorrelation is what closes it. Run together and a partial result is unattributable.

**Step 1a — paired twins.**
- Paired source / no-source examples sharing **identical** visit dates, count, coverage, masks, exposure.
- Source example → joint class/redshift target. No-source twin → low adequacy + broad redshift.
- **The "broad" target is the training z prior itself as a soft target** (cross-entropy against p_prior(z)), *not* uniform over the grid and *not* an entropy penalty. Uniform distorts posterior shape on real sources; a moment-matched max-ent target is ambiguous to implement. Write the exact distribution into the config.
- **Adequacy target for the twin is a soft label of 0.05**, not a hard 0 — a hard target saturates the sigmoid and stops gradient flow to the marginal cases that matter.
- Measure against the pass criteria. **Then** proceed.

**Step 1b — span decorrelation.**
- **Redshift-independent visit-*span* sampling**, not only visit count. Span is the stronger correlate (r=0.299 vs 0.223).
- **Scheme: reject-sampling against a fixed target span distribution.** Choose the target as the empirical span distribution of the *lowest* redshift decile (the achievable envelope); for each object, sample visit subsets and accept with probability ∝ target(span)/observed(span | z-bin). Do **not** synthesise spans, which would break the spectra-to-time pairing. Record the acceptance rate — if it falls below ~0.2 in any z-bin, the sample is too depleted and the bin must be reweighted instead.
- Re-measure. The delta between 1a and 1b is the gated-timing contribution.

**Both steps:**
- **Timing-only diagnostic model** — input is **visit times only** (no counts, no exposures, no coverage), so that it isolates schedule shape rather than depth.
- Repeated random visit-subset tests; explicit paired source-vs-no-source evaluation.

**Known limitation to record:** the paired loss suppresses *ungated* timing use. A model
can still learn "if adequacy is high, use span to pick z" — the twin never punishes
that, because the twin genuinely should be broad. Span-matched sampling is what closes
that path. Note that span ∝ (1+z) is **partly real physics**: time dilation is legitimate
evidence when read as spectral evolution against observer elapsed time. Decorrelating
span removes the bare-fact route while leaving the evolution route intact.

**Pass criteria — pre-declared, falsifiable, with margins and seed counts.** The
previous phrasing ("falls toward ~6%", "widens from +0.6 pp") gave a direction with no
threshold, which violates Rules 1, 2 and 4.

| # | Criterion | Threshold | Test | Seeds |
|---|---|---|---|---|
| 1 | No-source near-z rate (within \|Δz\|<0.1) | **≤ 12.0%** (upper 95% CI bound below 12%; floor is 6.2%) | paired vs the 33.4% baseline on the same 500 objects | 5 |
| 2 | Timing-only baseline | **within 3 pp of chance**, upper 95% bound below chance+3pp | one-sample vs chance | 5 |
| 3 | Paired source-vs-no-source gap | **≥ +5.0 pp**, lower 95% CI bound above 0 | exact paired test, same objects | 5 |
| 4 | Clean-source control (N6) | must **not** degrade beyond the class-accuracy margin 0.010 | paired | 5 |

Report criterion 3 **stratified by z**: it should open most at low-to-mid z and stay
near zero at z≳2 where the source genuinely isn't visible. **A uniform improvement
across z is a failure signal, not a success** — it means something other than spectral
sensitivity changed.

**Stop when:** all four criteria pass with paired CIs at ≥5 seeds, reported in the
Rule-4 format (`TO SETTLE` / `INTERIM` lines on anything underpowered).

### Change 2 — port ONIR, with the atlas fixes, in the right order

**Aggregation goes before coverage extension** (A4) — mechanistically motivated, and it
costs nothing to keep the order. It is *not* backed by a measured effect size; do not
defend it as "not optional" to a referee.

**Land as four separately measured steps.** Six fixes in one step violates Rule 8: three
have independent RESOLVED effects (logcov +0.032; support+logcov +0.066; clean templates
+0.064) and two are adopted on **cost** rather than accuracy — run together, the
cost-justified changes will absorb credit or blame from the accuracy-justified ones.

- **2a — log-coverage aggregation** with **fractional** coverage (A5), replacing `mean` over fixed F. Measure. *(expected: +0.032 f1)*
- **2b — clean class-mean templates**, `min_train_windows: 25`, back off to `evidence_group`-shared prototype, **never fabricate** (A11, A12). Measure. *(expected: +0.064 f1, +0.164 rare-class recall)*
- **2c — symmetric NCC (A15) + explicit phase-support weight (P5), jointly.** These are the RESOLVED +0.066 pairing and each is UNDERPOWERED alone, so measure them together and say so. Measure.
- **2d — wire dead metadata (A8) + per-feature asymmetric windows (A7) + extend to 23 features (A3), as one batch.** Adopt on **cost and coverage**, with **no accuracy claim**. Report compute change (−28% on the existing 15; +10.7% at 23) and the coverage table, not a metric win.

**Comparison ladder** (all with matched no-source controls and identical dates):
- ONIR spectral-only, no phase or dates
- ONIR + temporal compatibility (time enters *only* through measured spectral evolution)
- timing-only diagnostic (control)
- **ONIR with random-position anchors** (control — the arm that distinguishes "the anchors work" from "any 15 windows work")
- bank init: clean vs medoid vs random profiles

**Stop when:** ONIR spectral-only beats the phase-neutral CNN on redshift with the
no-source gap intact.

### Change 3 — encoder, as an isolated change

Retrain the **current** stem under the new uncertainty definition first and accept that
baseline. *Then* run stem-vs-stem with a random-init control, and **gate adoption of
the dilated stem on OpenUniverse** (E3). Do not bundle with the uncertainty change,
MAE pretraining, or the phase contract.

---

## 4. The strider tree

The current pilot layout is good. Two additions the review found missing from **both**
proposed trees in the design docs:

```
strider/
├── README.md
├── pyproject.toml
├── src/strider/
│   ├── cli.py
│   ├── config.py            # ONE typed tree, strict, snapshot into every run dir
│   ├── types.py             # SpectrumSeries, PhaseContext, PreparedBatch, Prediction
│   ├── data/
│   │   ├── prepare.py       # FITS -> native-bin store
│   │   ├── dataset.py       # the single reader for every view
│   │   └── noise.py         # <-- MUST BE A NAMED TOP-LEVEL FILE
│   ├── atlas/
│   │   └── onir_features.py # <-- the rest wavelengths IN SOURCE, one line + citation each
│   ├── model/
│   │   ├── encoder.py       # no phase argument in the signature at all
│   │   ├── redshift_scan.py # gather -> match -> aggregate -> combine
│   │   ├── signatures.py
│   │   └── validity.py      # the ONLY consumer of absolute scale
│   ├── training/
│   ├── evaluation/
│   └── model_file.py        # save / load / inspect, self-describing
├── configs/
├── slurm/                   # <-- 61 scripts today have no home in either proposed tree
├── experiments/
├── tests/
└── docs/
```

**Why `noise.py` and `onir_features.py` are named top-level files:** neither proposed
tree in the design documents contains any file whose name includes "noise", while the
entire scientific programme is about the noise model; and the 15 rest wavelengths are
~15 numbers with literature provenance that belong in **source**, with the atlas NPZ
derived from them and checksummed against them.

**Boundary rules worth enforcing in CI:**
- Only `phase_context` divides by (1+z). Grep-enforceable; the expression currently appears in three files.
- `redshift_scan.gather` returns a support **weight**, not just a boolean.
- Exactly one NCC implementation.
- `validity.py` is the only module that sees absolute amplitude.

---

## 5. Local Sundial gate sequence (must pass before NERSC)

| # | Gate | Pass criterion |
|---|---|---|
| 0 | Clear the import landmine | `import strider` from outside the repo resolves to the intended package or fails |
| 1 | Native extraction | `dataset_info.json` present; converter **prints actual bin counts and bytes by field** |
| 2 | Exact FITS round trip | Bitwise at FITS dtype; `native_bin_count == NBIN_LAM`; offsets provably cannot address another epoch's bins |
| 3 | Full 90-pair run | Completes; peak RSS recorded; no file-handle growth |
| 4 | **Resampling parity, old vs new** | Exactly equal, or every difference explained by a named tested correction |
| 5 | **Frozen-model prediction parity** | Tolerances agreed **in writing first** |
| 6 | Config-trap regression | Unread-field set empty; misspelled key raises |
| 7 | Repeatable generation | Bit-identical for the same 5-part key; generation strictly precedes resampling |
| 8 | **DataLoader worker parity** | `num_workers` 0 vs 4 bit-identical on a ≥2-shard store — the cheapest local test for the pilot's 160× slowdown |
| 9 | Evaluation-cache rejection | Training command refuses it; no config/env/flag can flip the loader |
| 10 | Generator control | Deterministic-seed equivalence: 0/100 discordant ⇒ discordance ≤2.95%. Plus **negative controls NC1 (flattened envelope, must collapse) and NC2 (shuffled)** — a control with no negative arm is not a control |
| 11 | Small end-to-end training | Finite with a 1-real-63-padded fixture; **anti-uniformity** asserted, not just finiteness |
| 12 | Exact resume | Bit-identical state dict **and** next batch |
| 13 | Export / reload / predict | `model-info` runs without importing torch |

**Gates 4 and 5 are the rewrite-vs-refactor decision.** If training and inference
preprocessing cannot be expressed as one shared function without moving the frozen
checkpoint's outputs, the coupling is load-bearing and a clean-room rebuild is
justified. Two days, locally, and it fails informatively in both directions.

---

## 6. First-week NERSC sequence

**CPU (days 1–2):** locked env from a committed lockfile → full test suite on the
cluster (fix the collection ImportError first — `pytest tests` currently exits non-zero
regardless of the other 815 tests) → import-origin check → convert 5–10k objects →
split/leakage audit → storage bake-off on **disjoint shards with rotated order**
(shared caches cannot be cleared) recording fd count and **Lustre metadata ops/sec, not
just throughput** → near-full conversion.

**GPU (days 3–5), all gated on `smoke_passed`:** tiny smoke on **debug** QOS →
**exact-resume gate at `num_workers=8`** → worker-parity at scale → 15-epoch smoke
(house rule) → worker sweep → first real comparison on **1 GPU, not 4**.

**Highest-value single NERSC test (~2 min):** does the all-masked attention row NaN on
**CUDA**? If yes, every reported eval metric is chance-level. If no, the bug is latent.
This reconciles the audit's "live NaN bug" with the reported macro-F1 of 0.779.

---

## 7. The three experiments that gate everything else

| | Experiment | Cost | Decides |
|---|---|---|---|
| **X1** | **De-redshift alignment of the log-σ residual.** Resample each object's residual log-FLAMERR onto a uniform log-λ grid, then onto a common **rest-frame** grid using true z. Compare alignment rest-frame vs observer-frame; SIM_FLAM residuals as positive control, z-shuffled as negative. **Two mandatory fixes — see below.** | ½ day, no GPU | **Whether the high-z channel is a simulation artefact or a real instrumental effect** — i.e. whether the whitened retrain happens at all. Nothing expensive should precede it. |
| **X2** | **Rebuild the shipped checkpoint from source.** Rebuild `strider_15class.pt` from a repository configuration at a known commit; diff the state dict. | 1 day, no GPU | Rewrite vs refactor, empirically. Also recovers (or definitively loses) provenance given `training_config_hash=''` and `36717bb-dirty`. Start regardless of sequence. |
| **X3** | **Real-data ONIR feature ablation with an unnamed-position sham.** Ablate each named feature on Hourglass2; measure per-class z and class change against a sham removing equal template power at unnamed positions. | 2 days, GPU | Whether the named vocabulary does physical work **on real data** — the paper's central interpretability claim. Outranks further synthetic seeds. |

### X1 is confounded as originally specified — two fixes are mandatory

An independent check found the design **well-posed but confounded**. It is not circular
(de-redshifting *destroys* alignment if the pattern is observer-locked and *creates* it
if rest-locked, so the observer arm is a genuine alternative). But:

**Confound 1 — asymmetric resampling.** The rest-frame arm applies a per-object
interpolation whose kernel depends on z; the observer arm applies none. Any difference
in sharpness is partly an interpolation artefact — the same failure §10.1 already
records, merely relocated. **Fix:** pass the observer arm through an *identical*
resample (shift by a z′ drawn from the same z distribution, map back to a common
observer grid) so both arms carry the same interpolation. Report the z-shuffled arm as
the floor for both.

**Confound 2 — N8 will dominate the statistic and give the right answer for the wrong
reason.** The NON1ASED red cutoff at **rest 10331.3 Å** is a rest-frame-locked,
simulation-origin, enormous FLAMERR feature present in 20–33% of CCSN epochs. Run X1 on
a mixed sample and it *will* find rest-frame alignment — the template cutoff, not the
grid beat. The conclusion "simulation artefact" would be correct but unearned, and the
whitening decision would rest on a mechanism the test never probed. **Fix:** primary arm
is **SALT3 Ia only** (0.1% affected), with CCSN as a labelled secondary arm and the
>1000×-median bins explicitly masked. Disagreement between the arms is itself a result.

**Third, narrower:** rest-frame alignment is not exclusive to a grid beat — a residual
source term or the template-library edge would also be rest-locked. All three are
simulation artefacts, so the *decision* (artefact vs instrument) survives, but **stop
attributing X1's outcome to the beat specifically.** Pair it with simulation request #2
(the ×0/×0.5/×1/×2 source-Poisson multiplier), which discriminates where X1 cannot.

**Stop doing:** any work premised on source-Poisson whitening; the coadd-only
architecture; scheduling the whitened retrain before X1.

> **Note on labels:** experiments in this section are **X1/X2/X3**; **E1–E9** are encoder
> rows in §2.3. An earlier draft used E1/E3 for both, which made §3's "gate adoption on
> OpenUniverse (E3)" ambiguous. §3 Change 3 refers to the **encoder** row E3.

---

## 8. Remaining simulation requests

Local data has **retired** three previously-deferred items: native-bin covariance
(zero), count-space necessity (Gaussian adequate), and whether 1/A is constructible
(yes, locally).

Still required, in priority order:

1. **CCSN template coverage** — extend the NON1ASED library past 10331 Å rest, or emit an explicit out-of-model-coverage flag instead of a huge `FLAMERR`. **Highest priority: every CCSN result below z=1 is confounded.**
2. **Source-Poisson multiplier arm (×0, ×0.5, ×1, ×2)**, everything else fixed. Decisive for the mechanism: if the leak survives ×0, source-Poisson is refuted outright — one simulation arm versus five training pilots.
3. Per-bin `VAR_SEARCH_SKY` / `VAR_TEMPLATE` / `VAR_READ` / `VAR_SOURCE` separately — needed for component ablation, **not** for background-only weighting.
4. Per-visit sky/zeropoint provenance (SIMLIB row) or a `noise_condition_id`. Two exposure conditions locally is not enough to validate a shared A/B table at survey scale.
5. An alternative ETC/extraction arm with a stated covariance property — the covariance null applies to *this* SNANA config only.
6. Post-LSF noiseless flux on native bins. **Downgraded**: measured cost is ≤0.05σ. But it **blocks clean-target MAE**, which would otherwise train an LSF deconvolution.

**External validation:** OpenUniverse (the E3 gate for the dilated stem), SIRAH, the
held-out noise construction. Note that held-out-noise transfer **preserved accuracy
while destroying calibration** (coverage-68 0.527; pure-noise lock 15× baseline) — so a
held-out-noise acceptance test that checks only accuracy will pass a model that has
stopped knowing what it doesn't know.

---

## 9. Risk table, ordered by consequence

| # | Risk | Consequence | Status |
|---|---|---|---|
| 1 | **The high-z redshift channel may be a simulation artefact.** Mechanism unknown; source-Poisson and epoch-common both refuted | If confirmed, high-z numbers measure the simulator, not physics, and real-Roman performance is unknown | **X1 settles it.** Highest priority |
| 2 | **Zero-support features score exactly 0.0** = "indifferent", better than any anti-correlated match, summed over epochs | A structural high-z attractor **independent** of the variance leak. Two leaks pointing the same way; fixing one may not move the answer, which will be misread | Confirmed; fix is A5 |
| 3 | **P(class) is confidently wrong on pure noise** (0.42–0.94), architecture-invariant | Live on the SMDC deployment. External, reputational | Confirmed; only fix is the external gate (V2) |
| 4 | **Calibration fitted on test**, shipped marked `provisional` | Every quoted ECE and coverage number is optimistic by an unknown amount, in the artefact users have | Confirmed |
| 5 | **A dead config block impersonates the live one.** `configs/strider_15class.yaml`'s `feature_head` (128/3/4) is *v2* and inert — the checkpoint's `model_config.feature_head` records the same values, so file and checkpoint agree. The live detector numbers (500/8/7) come from `configs/train_15class.yaml:8,11,14` and are **correct**. The defect is that the dead block sits in the file an author reads *and* is validated at `config.py:324-335`. Plus `training_config_hash=''`, commit `dirty` | An author writing methods from the named config states four wrong numbers. **Earlier drafts said "the run cannot be repeated" — that overstates it: the detector spec exists and is right.** Provenance of the *run* remains unrecoverable | **X2 settles the run provenance** |
| 6 | **Cadence supplies most of the pilot's redshift** (33.4% → 6.2%) | Any ONIR comparison run now measures the wrong thing | Change 1 |
| 7 | **CCSN template cutoff** feature-free z readout at NMAD 0.0029 | Every CCSN result below z=1 confounded | Scoped; simulation request #1 |
| 8 | Silent phase clamping (18.9%, low-z loaded); `signature_active_fraction` pinned near 1.0 by construction | The health metric cannot report the failure it exists to catch | Confirmed; fix is P5/A14 |
| 9 | Partial-support NCC asymmetry (√ of in-band power); two conventions coexist | Class-dependent bias, worst exactly where features leave the band | Confirmed; fix is A15 |
| 10 | **Acceptance tests that pass on live bugs** ("assert finite"); positive controls with no negative control | The review process itself does not catch this class — the fifth data-contract failure in this project | Systemic |
| 11 | Coadd complexity for a low-S/N-only, coverage-costing gain | Migration risk for −20% in one regime | Deprioritise below 1–9 |

---

## 10. Explicitly unresolved — do not average these away

1. **Artefact or instrument?** Suggestive but not decisive evidence that the σ residual is rest-frame-locked (§1 tags this UNRESOLVED and that is the operative tag); a conflicting alignment test was contaminated by a non-uniform log-λ grid. **E1.**
2. ~~The kNN result is unverified and withdrawn.~~ **RESOLVED — withdrawal reversed.** The named controls were run (§1): blocking on `SIM_LIBID` and HEAD file leaves it intact; truth z never enters the predictor; grid length and band edges are constant by construction; the target-permutation null is 0.114–0.126; the `SIM_TEMPLATE_INDEX` block is degenerate because all SALT3 Ia share one value, which dissolves rather than fails the confound. The specificity profile (redshift skill +0.978 vs host mass +0.008, MWEBV +0.009, and +0.922 at fixed brightness) additionally rules out host light and distance-modulus routes. **What remains open is the channel's *origin*, not its existence.**
3. **Do the named anchors earn their keep on real data?** RESOLVED in synthetic; **X3** settles it where it matters.
4. **Rewrite vs refactor.** Amended to "extract, then decide", gated on **X2** rather than on preference.
5. **The real phase-evolution rate** in coadd-blur units — the ~20-day safe span is derived from a literature velocity-decline estimate, not measured on this data.
6. **Coadd form** (per-epoch channel vs object-level branch) — both UNDERPOWERED, evidence weakly against the documented preference.
7. **The dilated stem on OpenUniverse** — the single decisive test before adoption.

---

## 11. Corrections made during this review

Recorded so they are not silently re-inherited:

| Claim | Correction |
|---|---|
| "Fe II 4555 clips at z=2.93, docs are wrong" | **Wrong, and the first correction was also wrong.** The window is **15 patches = 120 bins, half-width 60 bins** (verified three ways: `edna_detector.py:56` comment; `bank_metadata['window_half_bins']=60`; `prototype_windows` shape `(15,16,15,8,121)` = 2·60+1). Not 56. Half-width 60 bins = **±15,404 km/s**; Fe II 4555 full-window clip **z=2.754**. Docs' z≈2.76 is right. But Ca II NIR clips **first at z=0.99**, and at z=0 only 2/15 features are in band — the real story is a two-ended staircase |
| "d log σ / d log F = −0.45, source-Poisson refuted by sign" | **Confounded.** That regression was across wavelength within an epoch, where A(λ) is anti-correlated with the source. Correct per-bin across-epoch value is **+0.001**. Source-Poisson is **negligible**, not wrong-signed |
| "Named ONIR positions don't earn their keep" (2 seeds) | **Power failure.** At 20 seeds, paired: −0.205 F1, RESOLVED-WORSE |
| "The truth-phase seam is worth zero" | **UNDERPOWERED** at decision-relevant margins (needs 6 seeds at δ=0.02, 24 at δ=0.01). The phase redesign is justified on **deployability**, which needs no equivalence claim |
| "Timing contributes 8% of the information" | **The 8.3% is the chance floor.** Temporal-only reached 8.6%; ~66,000 objects would be needed to resolve that. Correct claim is an upper bound: <1.5 pp |
| "Timing supplies 98% of the redshift signal" | **Invalid decomposition** across three non-additive interventions, one out-of-distribution. Correct claim: timing **dominates**, a smaller spectral contribution is measurable |
| "The pilot's HDF5 slowdown is inherited file handles" | **Wrong.** The pilot already opens lazily (`dataset.py:289`) and clears in `__getstate__`. Cause is open |
| "The spectral route contributes nothing" | **False.** The phase-neutral model gives 14.8% vs 5.6% no-source, +9.2 pp, p=2.2e-6. Weak but demonstrably nonzero. My unpaired significance test was also wrong — paired is required |
| "Epoch-common noise caps the coadd gain" | **Refuted.** +8e-5 over 543,414 pairs |

**Corrections from the independent fact-check of this document (second review pass):**

| Claim | Correction |
|---|---|
| Section 11's own correction, "7 patches = **56 bins**" | **Also wrong.** Window is 15 patches = **120 bins, half-width 60** - verified three ways. +/-15,404 km/s; Fe II 4555 clip z=2.754. A correction that fixed one unit error introduced another |
| "28.5% of the bank is fabricated or a single noisy window" | **Labels inverted.** 13.4% under-populated, 15.1% low-N *means* (the cleanest class), **71.5% single-window medoids** |
| KN "157/240 fabricated (65%)" | **Disputed** across three computations (157/87/199). 10 of 16 phase bins have zero supported cells. Pin before quoting |
| `signature_active_fraction` "pinned near 1.0" | Reads **0.688-0.798**; denominator is K=8 slots, not cells |
| Stem RF "18 bins = 115-277 A" | **22 bins = 141-339 A**. State RF in bins (z-invariant): 22 vs 70 for an 18,000 km/s trough |
| "The run cannot be repeated" (Risk 5) | Overstated - the detector spec exists and is correct in `train_15class.yaml`. The defect is a dead block that *looks* live |
| "7 distinct target_classes sets" | **4** under the expansion that gives 66/105 |
| X1 (was E1) as specified | **Confounded twice** - asymmetric resampling, and N8's CCSN template cutoff would dominate the statistic, making "artefact" correct for the wrong reason |
| E1/E3 used for both encoder rows and experiments | Experiments renamed **X1/X2/X3** |
| Section 2.6 stated without seed counts | **n=2, no CIs** - flagged inline |


**Corrections made after the first draft of this document, in its own review:**

| Claim in draft 1 | Correction |
|---|---|
| A4 effect sizes (+20.1% / +0.044 / +46.6%) making the step order "not optional" | **Untraceable in any artifact**; "20.1%" collides with an unrelated figure. Order retained on the VERIFIED mechanism only |
| A6 `overlap_norm` repeat test tagged RESOLVED | **UNDERPOWERED** (n_req=22). Root cause at `edna_detector.py:262` is VERIFIED and carries the recommendation alone |
| P5 phase-support fix tagged RESOLVED at "+0.084 low-z accuracy" | Fix **alone is UNDERPOWERED** (n_req=4); RESOLVED only jointly with A5. The "+0.084" figure could not be located |
| "18.9% of the (Δt,z) plane clamps" | **UNVERIFIED** — may describe a drawn rectangle, not the empirical distribution. Use the sourced 20.1%-of-populated-cells figure |
| §2.4 stated without a power caveat | The ONIR study's **own positive control failed** (n_req=13). No per-feature claim is admissible at 20 seeds |
| C8 justified by flat span / time-grouping results | **Wrong knob.** Span at fixed evolution rate tests sampling extent, not blur. The `vel_scale` arms show coadd degrades ~2× faster than per-epoch, and the toy understates the real rate ~2× |
| kNN result withdrawn as unverified | **Withdrawal reversed** — controls run and passed; specificity profile added |
| Change 2 as one six-part step | **Split into 2a–2d**, each measured. Violated Rule 8 as written |
| Metric contract (M1–M4) had no step | Added as **Change 0**, landing alone and first |
| Change 1 pass criteria stated as directions | Replaced with thresholds, tests and seed counts |

---

## 12. Standing methodological rules

1. **Minimum 5 seeds** for any seed-variance claim, 10 for a headline. Report mean ± SD over seeds, paired ΔCI, margin, verdict.
2. **Pre-declare equivalence margins** before looking: class accuracy 0.010; macro-F1 0.020; NMAD 10% relative (floor 0.0005); median|Δz| 10%; outlier fraction 0.020 (0.005 for the gold-Ia sample); coverage-68 0.030; CRPS 5%.
3. **Report φ_seed**, the seed-limited fraction of the variance. If φ_seed > 0.5, adding objects is wasted compute.
4. **UNDERPOWERED is a first-class verdict** and must carry `TO SETTLE` (the N) and `INTERIM` (what to do meanwhile). "No significant difference" is forbidden.
5. **Pair everything that can be paired** — same objects, same seeds, same `realization_number`.
6. **n_seed = 1 is admissible only** when the effect exceeds ~10× a σ_seed measured elsewhere on the same system.
7. **Deterministic comparisons** need no power analysis for *existence* claims and always need one for *magnitude* claims.
8. **Never bundle.** The metric change (M1–M4) is free and should land alone, first.
9. **Synthetic numbers test mechanism, not supernova accuracy.** Never quote one as an astrophysical result.
