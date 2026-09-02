# STRIDER — project setup, local and NERSC

**Date:** 2026-08-02
**Purpose:** the concrete file layout, config discipline, documentation path and NERSC
setup, folding in the review findings. Companion to
`consolidated_review_2026_08_02.md` (the *what* and *why*); this is the *where*.

---

## 1. What is already right — do not churn it

Measured against the v2 sprawl this is in good shape and most of it should be left alone.

| | v2 | v3 today |
|---|---|---|
| core Python | 15,252 LOC | **3,694 LOC across 35 files** |
| one-off scripts/diagnostics | 78,956 LOC | none yet — keep it that way |
| near-duplicate packages | 4 (`strider`, `strider-clean`, `strider-public`, `strider_smdc`) | 1 |
| entry points | 272 `__main__` blocks | 1 (`cli.py`) |
| config files | 79 (51% from dead lineages) | 15, all live |

Already fixed relative to the review's findings:

- **Per-feature window geometry is wired.** `configs/onir_features.yaml` carries
  `half_width_kms` per feature (20,000 for Ca II H&K, 14,000 for S II, …). In v2 this
  field was parsed and never read (A7/A8). Done.
- **`target_classes` is deliberately absent** from the atlas, with the reason recorded in
  the file header. The review found hard class-masking was the single worst arm tested
  (F1 −0.314). Correctly avoided.
- **`src/` layout** — kills the cwd-shadowing failure that made `import strider` resolve
  to the wrong package in v2.
- **Separation of concerns** in `src/strider/{atlas,data,model,training,evaluation}` is
  already close to the target.

---

## 2. Gaps to close, in priority order

| # | Gap | Why it matters | Fix |
|---|---|---|---|
| **G1** | **No exact resume.** `grep -E "resume\|optimizer.state_dict\|rng_state" training/trainer.py` returns nothing | v2's published model came from a chain of `--init-from-checkpoint` runs, so its effective LR schedule is unrecoverable from any config. This is a paper-methods problem, not just engineering | `training/resume.py`; gate on **bit-identical** state dict *and* next batch |
| **G2** | **Trial-redshift grid is 60 points** (`configs/local_pilot.yaml:35-37`) → rounding floor of median \|Δz\| = **0.0125** | Every redshift metric from a NERSC run inherits this floor. The clean arm is currently a null instrument | ~500 points + sub-bin quantile interpolation |
| **G3** | **Noise generation is spread across 9 files**, with no module named for it | The most contested scientific object in the programme should be findable by `ls`. Neither proposed tree in the design docs had a file whose name contained "noise" | `data/noise.py` — one home for generation, the 5-part RNG key, and the noise-construction registry |
| **G4** | **No `slurm/`** | 100% of the science compute will run there and it has no home | `slurm/` with one script per *command*, not per experiment |
| **G5** | **Adequacy logic spread across 8 files** | Validity must be the *only* consumer of absolute scale, and that is enforceable only if it is one module | `model/validity.py` |
| **G6** | **No self-describing model package** | v2 shipped `training_config_hash=''` and commit `36717bb-dirty` | `model_file.py` + the package spec in §5 |
| **G7** | **No calibration split** | v2 fitted temperature on the test set and reported ECE on the same rows | 5-way split, §4 |
| **G8** | **Docs are results-only** | An astronomer cannot currently follow one object through the system | §6 reading path |
| **G9** | **No `experiments/`** | One-off studies otherwise land in `src/` and become the next 78k LOC | `experiments/`, dated, read-only after landing |

---

## 3. Target local tree

Additions marked `NEW`. Everything unmarked exists and stays.

```
strider/
├── README.md                        # what it is + one prediction in 10 lines
├── pyproject.toml
├── src/strider/
│   ├── cli.py                       # the only entry point
│   ├── config.py                    # one typed tree, strict, snapshot per run
│   ├── types.py               NEW   # SpectrumSeries, PhaseContext, Prediction
│   ├── atlas/
│   │   ├── catalog.py               # reads configs/onir_features.yaml
│   │   ├── build.py                 # bank construction
│   │   └── bank.py
│   ├── data/
│   │   ├── snana.py                 # the ONLY format-specific file
│   │   ├── prepare.py               # FITS -> native-bin store
│   │   ├── noise.py           NEW   # generation + 5-part RNG key + registry
│   │   ├── records.py
│   │   └── dataset.py               # one reader for every view
│   ├── model/
│   │   ├── spectral_encoder.py
│   │   ├── spectral_tokens.py
│   │   ├── redshift_scan.py         # gather -> match -> aggregate
│   │   ├── encoded_onir.py
│   │   ├── phase.py                 # the ONLY module that divides by (1+z)
│   │   ├── temporal.py
│   │   ├── validity.py        NEW   # the ONLY consumer of absolute scale
│   │   └── strider.py               # assembly
│   ├── training/
│   │   ├── trainer.py
│   │   ├── resume.py          NEW   # G1
│   │   ├── losses.py
│   │   └── device.py
│   ├── evaluation/                  # already well factored - leave alone
│   └── model_file.py          NEW   # save / load / inspect / verify
├── configs/
│   ├── onir_features.yaml
│   ├── local_pilot.yaml
│   └── experiments/                 # one self-contained YAML per run
├── slurm/                     NEW
│   ├── requirements.lock            # pip freeze from the blessed env
│   ├── setup_env.sh                 # installs FROM the lock, never loose specs
│   ├── prepare.sbatch               # CPU
│   ├── smoke.sbatch                 # GPU, qos=debug HARD-CODED
│   ├── train.sbatch                 # GPU, 1 GPU unless DDP
│   └── evaluate.sbatch
├── experiments/               NEW   # dated one-offs, read-only after landing
├── tests/
├── docs/
└── tutorials/                 NEW   # notebooks (different CI rules than tests/)
```

**Four boundary rules, each grep-enforceable in CI:**

1. Only `model/phase.py` contains `(1 + z)` as a divisor. In v2 this expression appeared in three files.
2. `redshift_scan.gather` returns a support **weight**, not a boolean.
3. Exactly one NCC implementation. v2 had two conventions and shipped the biased one.
4. `model/validity.py` is the only module that sees pre-normalisation absolute scale.

---

## 4. Config and split discipline

**One config file per run. One `--config` flag.** v2's trap was two config files where the
one named in the sbatch script (`strider_15class.yaml`) declared `z_scan_n_bins: 128`,
`window_radius: 3`, `z_scan_dim: 4` while the model was 500/7/8 — and the dead block was
*validated* at `config.py:324-335`, so it looked live.

Three mechanisms make that class of trap structurally impossible:

1. **Strict parsing** — unknown section or key raises with a `difflib` suggestion. Never `continue`.
2. **`test_no_dead_config_fields`** — wrap the config in an access-tracking proxy during the smoke test and assert `cfg.unread_fields() == set()`. This is the test that catches *known keys that are never read*, which strictness alone does not.
3. **Resolved snapshot written before step 0** — full dataclass round-tripped to YAML (defaults included) into the run directory, plus `environment.json`. If the run directory exists with a different resolved config, refuse unless `--resume` (requires byte-identical) or `--force-new-run`.

**Splits — five-way, by underlying source, with two orthogonal held-out axes:**

| split | may touch | must never touch |
|---|---|---|
| TRAIN (0.70) | weights, bank construction | anything else |
| SELECT (0.10) | early stopping, checkpoint choice, all parameter studies, calibration *method* choice | never fits a deployed scalar |
| CONFIRM (0.05) | one re-ranking of the ≤3 finalists | anything else |
| CAL (0.07) | calibration coefficients, adequacy thresholds | must not influence checkpoint or method choice |
| TEST (0.08) | one frozen pass, once | everything |

Orthogonal: the **held-out noise construction** and **external domains** (OU, SIRAH)
appear only in TEST, with an automated check proving absence from the other four.
CAL needs ≥~2,400 objects to verify 68% coverage to ±2%.

**Split identity travels inside every predictions file** — not in the filename. That is
the structural enabler of v2's calibration leakage.

---

## 5. Model package

`strider model-info` must answer "what is this and can I trust it?" **without loading
weights or importing torch**.

```
model_id/
├── model_info.json          identity, class list, supported input range
├── config.resolved.yaml     THE FULL CONFIG TEXT, not a hash
├── provenance.json          git sha + clean/dirty + diff-if-dirty; dataset id +
│                            checksum; bank id + checksum; slurm job id;
│                            python/torch/numpy versions; ulimit -n
├── weights.safetensors
├── onir_features.yaml       the anchors as trained + sha256 of the source
├── bank_state.npz
├── wavelength_grid.npy, redshift_grid.npy
├── preprocessing.yaml       resample rule, normalisation, channels, masks
├── calibration.json         method, split fitted on, N, params - or "none"
├── metrics.json             headline numbers WITH split identity and N
├── example_input.npz, example_output.json
├── MODEL_CARD.md            what it is for, what it is NOT for, failure modes
└── SHA256SUMS
```

Two hard rules: **a dirty git tree emits its diff into `provenance.json` or the export
fails**; and the embedded config is the resolved **text**, because a hash tells you two
runs differ but not how.

---

## 6. Documentation path — the accessibility requirement

The primary maintainer is an astronomer learning the ML. Optimise for that.

**Day 1 (60 minutes, in order):**
1. `README.md` — what STRIDER does, one worked prediction. One page.
2. `docs/start-here.md` — the 10-line diagram: spectrum → tensor → scan → posterior → validity, **each box naming the file**.
3. `tutorials/classify-one-object.ipynb` — run it, see numbers.
4. `docs/glossary.md` — ONIR, rest phase, signature, bank, scan, adequacy, NMAD, coadd. Plain language before abbreviation.

**Week 1:**
5. `docs/how-strider-works.md` — **one object followed through every component with the tensor shape at each step.** This is the document that replaces oral history.
6. `docs/ml-choices.md` — per component: problem → intuition → choice → alternatives → **evidence** → limitation. Where "the profiles matter 20×, the exact positions are unresolved" lives.
7. `tutorials/train-tiny-model.ipynb` — 5 minutes on a laptop, end to end.
8. `docs/runbooks/{prepare,train,evaluate,export}.md` — copy-pasteable, with expected runtimes.

**Contributing:**
9. `docs/extending-strider.md` — where a new adapter / channel / loss / diagnostic goes, and the test each must add.
10. `docs/decisions/` — one short ADR per settled argument.

**Module docstring rule.** Every module states purpose, inputs, outputs, **frames and
units**, **prohibited information** (e.g. "must never see `SIM_FLAM` or truth z"), and
the next file to read. The prohibited-information line is what would have made the
variance-leak audit cheap.

---

## 7. Local gate sequence before NERSC

Ordered. Each has a pass criterion. Runtimes are laptop estimates.

| # | Gate | Pass criterion | Est. |
|---|---|---|---|
| 0 | Import origin | `import strider` from outside the repo resolves correctly or fails cleanly | 5 min |
| 1 | Native extraction, 15 classes | converter **prints actual bin counts and bytes by field** | 15–40 min |
| 2 | Exact FITS round trip | bitwise at FITS dtype; offsets provably cannot address another epoch's bins | 5 min |
| 3 | Config-trap regression | unread-field set empty; misspelled key raises | <1 min |
| 4 | Repeatable generation | bit-identical for the same 5-part key; **generation strictly precedes resampling** | 10 min |
| 5 | **Worker parity** | `num_workers` 0 vs 4 bit-identical on a **≥2-shard** store, no fd growth | 5 min |
| 6 | **Exact resume (G1)** | **bit-identical state dict AND next batch** | 30 min |
| 7 | Evaluation-cache rejection | training command refuses it; no config/env/flag can flip the loader | 5 min |
| 8 | Independent-error-pattern control | redshift follows the source, not the error pattern; **plus the flattened and shuffled negative controls, which must collapse** | 45 min |
| 9 | Tiny end-to-end of the **exact NERSC config** | runs; resolved config + environment written; checkpoint exports and reloads | 60 min |

Gate 5 is the cheapest local test for the pilot's 548 → 3.4 obj/s slowdown. Gate 8's
negative controls are non-negotiable — a control with no failing arm is not a control.

---

## 8. NERSC setup

### What changes from local

| | local | NERSC |
|---|---|---|
| store | one HDF5 file | **sharded**, ~200–500 shards (fd budget = workers × shards touched) |
| workers | 0 (Mac-specific finding) | **benchmark from scratch**; lazy per-worker handles guarded on `os.getpid()`; `persistent_workers=True` |
| env | conda | built **from `slurm/requirements.lock`**, never loose specifiers |
| provenance | git sha | `record_environment()` on every command |

`num_workers: 0` must not be copied to Perlmutter — with Lustre, a single reader process
will bottleneck. The pilot's cause is *not* inherited handles (it already opens lazily at
`dataset.py:344-347` and clears in `__getstate__`); repeated worker startup with
`persistent_workers=False` (`trainer.py:119`) and single-file contention are the live
suspects. Sharding plus persistent workers addresses both.

### First week

**CPU (days 1–2), `qos=regular` or `shared`:**

1. Build locked env; record it. **Torch version must be recorded** — the CPU-eval NaN in v2 was version-dependent.
2. Full test suite on the cluster.
3. Import-origin check from `$HOME`.
4. Convert 5–10k objects; **print bytes by field**.
5. Split + leakage audit: zero source overlap; held-out noise construction provably absent from TRAIN/SELECT/CONFIRM/CAL.
6. Storage bake-off on **disjoint shards with rotated order** (shared caches cannot be cleared). Record **Lustre metadata ops/sec and fd count**, not just throughput.
7. Near-full conversion. Pass criterion: `ls` on the dataset dir returns in <30 s.

**GPU (days 3–5), each gated on the previous:**

| # | Step | QOS | Pass criterion |
|---|---|---|---|
| G1 | tiny smoke, 2 epochs | **debug** | finite everything incl. a 1-real-63-padded fixture; **anti-uniformity asserted, not just finiteness**; touches `smoke_passed` |
| G2 | exact-resume gate at `num_workers=8` | **debug** | bit-identical state dict and next batch |
| G3 | worker parity at scale | **debug** | bit-identical; no fd growth over 200 steps |
| G4 | 15-epoch smoke (house rule) | debug/shared | loss decreases; **GPU utilisation ≥60%** — if lower, the data path is the bottleneck, stop and fix |
| G5 | worker sweep | shared | throughput plateaus; GPU wait <20% |
| G6 | first real run: encoded ONIR baseline | regular, **1 GPU** | see §9 |

**Slurm rules, from v2's measured mistakes:** `retrain_15class_regular_chain.sh` booked
**4 GPUs with no DDP** (~42 idle GPU-h/job) and `retrain_15class_debug.sh` used
**regular** QOS. So: one script per command; `smoke.sbatch` hard-codes `qos=debug`; a
guard refuses `train.sbatch` unless `smoke_passed` exists for that config; 1 GPU unless
DDP is actually detected.

**Highest-value single test on the cluster (~2 min):** does an all-masked attention row
NaN on **CUDA**? On CPU it does (eval mode only); on MPS it does not. If CUDA NaNs, every
v2 eval metric was chance-level. Two minutes settles it.

---

## 9. The first NERSC run — configuration and gates

Architecture: encoded ONIR baseline, **no temporal, no coadd, no new stem**.

Setup: full 15-class population; larger balanced or flat-redshift training set; generated
observations with varied noise constructions; matched source-free examples with identical
visit selections; generated **and** clean validation for checkpoint selection;
**≥5 seeds** (3 gives a 12.1× CI span on the seed SD, which cannot support a stability
gate; 5 gives 4.8×).

Evaluate every checkpoint on: clean · generated noisy · original FLAM · new reported-error
realisation · **independently selected error pattern** · source-free · residual · held-out
noise construction · OpenUniverse when available.

**Metrics — the corrected contract.** NMAD on **(1+z)-normalised** Δz, always paired with
outlier fraction; median |Δz|; **CRPS** for the posterior; class precision/recall/macro-F1
(not accuracy alone); adequacy false-accept/reject; interval coverage fitted on CAL only;
all stratified by redshift, S/N, epoch count and coverage. Raw Δz **only** for the
error-pattern attraction statistic.

**Pre-declare every gate threshold before the run.** Qualitative gates get argued after
results appear:

| gate | make it a number |
|---|---|
| noisy → clean | noisy/clean ratio below a stated value (currently 17×) |
| source vs source-free | paired gap ≥ N pp, lower CI > 0 |
| z follows source under changed error | attraction ≤ 10%, with flattened/shuffled controls collapsing |
| adequacy rejects source-free | ≤ X% above threshold |
| improvement follows source | monotone in S/N and coverage — **uniform improvement across strata is a failure signal** |
| seed stability | SD across seeds below the margin per metric |

Only after those pass, add components **one at a time**, each with its own controlled
comparison: observable-only temporal evolution → visit aggregation → coadded context →
attention/CNN comparison → MAE pretraining.

---

## 10. Immediate task list

| # | Task | Owner | Blocking |
|---|---|---|---|
| 1 | Exact resume + bit-identical gate (G1) | impl | NERSC |
| 2 | Trial-redshift grid to ~500 + sub-bin interpolation (G2) | impl | interpretable metrics |
| 3 | `data/noise.py`, `model/validity.py`, `model_file.py` (G3, G5, G6) | impl | maintainability |
| 4 | `slurm/` + `requirements.lock` + `record_environment()` (G4) | impl | NERSC |
| 5 | Five-way split + split-identity-in-file (G7) | impl | any calibration claim |
| 6 | Independent-error-pattern control **with negative controls** | impl | the run's core question |
| 7 | Pre-declare all gate thresholds, in writing | science | the run's interpretability |
| 8 | **X1** — de-redshift alignment, with the two fixes | science | interpretation of #6; runs in parallel |
| 9 | Docs day-1 path (§6 items 1–4) | docs | onboarding |
