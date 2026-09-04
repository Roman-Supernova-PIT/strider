# First STRIDER models on NERSC

> **Historical experiment record.** These first-model routes and proposed
> comparisons do not define the current reference candidate. See
> [`architecture.md`](architecture.md) and
> [`research_history.md`](research_history.md).

This page fixes the order of the first large runs. Each comparison changes one
information route, uses the same prepared objects and clean ONIR bank, and is
scored on the selection role before calibration or final testing.

## Shared data

| Role | Simulation production | Purpose |
|---|---|---|
| Training | ten-seed flat-redshift Hourglass2 | fit model parameters |
| Selection | blocks 1--8 of the new flat-redshift production | choose architecture and settings |
| Calibration | blocks 9--10 of the new flat-redshift production | fit probability and interval corrections after selection |
| Final test | Sundial | read only after the method is fixed |

Preparation writes native variable-length spectral arrays to HDF5 and object and
visit records to Parquet. Measured flux, reported uncertainty and clean simulated
flux are stored once. Fresh observations and source-free controls are generated
by the loader on native bins before resampling.

## First converged comparison

The laptop runs diagnose gross failures but do not select a final model after
five epochs. The first converged comparison therefore uses:

- `configs/nersc/spectral_20k.yaml`: 64-dimensional spectral tokens;
- `configs/nersc/spectral_20k_large.yaml`: 128-dimensional spectral tokens;
- 20,000 training, 5,000 selection and 2,000 calibration objects;
- 30 epochs with early stopping after six non-improving epochs; and
- identical data, ONIR profiles, noise generation and losses in both runs.

NERSC `debug` checks one epoch and timing only. The converged fits use the shared
GPU queue.

## Model A: spectral shape

Configuration: `configs/nersc/spectral.yaml`

- shared encoder for every visit;
- per-visit centring and scaling for the ONIR shape comparison;
- 15 named rest-wavelength regions;
- 500 trial redshifts uniform in `log(1+z)` from 0 to 3;
- clean phase-neutral ONIR bank;
- visit mean rather than visit sum;
- no temporal branch;
- separate evidence-sufficiency result using background-scaled flux.

This is the first scientific baseline. It asks how far spectral shape and named
feature displacement can go without truth-derived phase or observing-schedule
information.

## Model B: background-scaled spectra

Configuration: `configs/nersc/spectral_scaled.yaml`

Model B changes only the second normalization. The loader still divides each
visit by one scalar background-noise scale, but the encoder retains the resulting
amplitude. This tests whether direct S/N information improves weak spectra and
whether it introduces an undesirable brightness route.

Choose between A and B only after comparing source-bearing, source-free,
residual, original, generated and reported-error views. A lower median redshift
error does not count as an improvement if source-free probabilities sharpen or
performance follows brightness rather than spectral structure.

## Model C: temporal evolution

Configuration: `configs/nersc/temporal.yaml`

Train this only after selecting A or B and rerunning the timing-only comparison
on the complete selection role. The temporal contribution starts at zero and
uses measured changes between visit representations with candidate rest-frame
intervals. It must improve class discrimination or reject redshift alternatives
without creating evidence from dates alone.

## Auxiliary phase comparison

Configuration: `configs/nersc/factored_phase_20k.yaml`

This extends the factored ONIR model with a class-conditioned phase
distribution from -20 to +50 days. Simulation phase is a loss target only. The
head does not change the class-redshift logits, so the first result asks whether
phase is measurable before using it as evidence. Report phase error, ordering
accuracy and 68% interval coverage for Ia and every other class. Also report the
largest phase-bin probability on source-free inputs.

The -20 to +50 day range is the common interval used to supervise phase across
classes. It is intentionally narrower than the -20 to +80 day interval used to
average the offline phase-neutral ONIR profiles.

## Full-spectrum context comparison

Configuration: `configs/nersc/factored_context_20k.yaml`

This adds a mask-aware CNN and two spectral-attention blocks over the complete
observed spectrum. It produces class context only; the contribution is constant
across trial redshift. The comparison tests whether the broader spectral shape
capacity used by STRIDER 2 improves Ia classification while ONIR continues to
set the redshift structure.

Compare it directly with `factored_20k` using the same objects and seed. Require
better selection-set Ia F1 or class probability scores, no new confidence on
source-free inputs, and unchanged redshift posterior shape when the class is
held fixed. Benchmark batch size before training because the additional
attention raises memory use.

## Coadded redshift comparison

Run this after the matched factored models, not as part of the phase-head run.
Coadd all measured visits in the observer frame and compare them with training
objects coadded by the same rule. The result is a broad class-redshift proposal
that may retain persistent features when individual visits have little signal.

Do not multiply its probabilities by the per-visit posterior as though they
were independent. First report the proposal width and the fraction of true
redshifts it contains. The detailed ONIR scan must remain available when the
proposal is broad or wrong. Source-free coadds must not produce narrow ranges.

## Settings for the first comparison

| Setting | First value | Why it is held fixed |
|---|---:|---|
| redshift bins | 500 | already resolves the scan more finely than the expected posterior width while retaining practical cost |
| token dimension | 64 and 128 | the only architecture difference between the two runs |
| profiles per class-region cell | 3 | keeps bank support and compute manageable |
| dropout | 0.10 | conventional starting regularization |
| learning rate | 0.001 | tested local starting value with AdamW |
| warmup | 3 epochs | protects a from-scratch encoder at the start of training |
| maximum epochs | 30 | upper bound controlled by selection loss and early stopping |
| batch size | 16 | passed the Perlmutter model-step test; batch 32 exhausted one A100 for the factored model |

These values are not described as optimal. Changing them during the capacity
comparison would make the result uninterpretable.

## First parameter study

After choosing normalization and before adding temporal evolution, vary parameters
in this order:

1. learning rate: `0.0003`, `0.001`, `0.003`;
2. token dimension: `16`, `32`;
3. ONIR profiles per cell: `3`, `5`;
4. dropout: `0.05`, `0.10`, `0.20`;
5. evidence-sufficiency loss weight: `0.25`, `0.5`, `1.0`.

Use short selection runs to remove clearly poor values, then repeat finalists
with at least five training seeds. Do not run the full Cartesian product.

## Required report

Every view and redshift interval reports:

- object count;
- class accuracy, macro F1, balanced accuracy, Ia precision, Ia recall and Ia F1;
- class log loss and Brier score;
- signed median raw `delta_z` and its 16th and 84th percentiles;
- median absolute raw `delta_z` and its full standard deviation;
- fractions with absolute `delta_z` above 0.05 and 0.10;
- 68% interval coverage and width;
- largest joint probability and redshift information gain;
- evidence sufficiency; and
- phase error, ordering accuracy and interval coverage when a phase head is present;
- results versus visit count and measured signal strength.

The full 0--3 range is always written. Use `z < 1.4` to confirm that the
source-driven v2 result is retained, then use finer bins through `z=2.2` to
measure where the new model loses source information. No validity boundary is
assigned before the source-removal and mismatched-noise checks are complete.

## Noise-robust acceptance criteria

Record these before comparing the final models:

1. replacing the wavelength-dependent noise pattern must not pull redshift
   toward the replacement source;
2. source-free inputs must have low evidence sufficiency and broad class-redshift
   probabilities;
3. clean-signal and generated-observation performance in the source-supported
   range must be retained;
4. one noise construction remains absent from training and is used only for
   evaluation; and
5. worse raw high-redshift precision is expected if the old precision came from
   simulated noise structure. It is not a regression when uncertainty and
   evidence sufficiency correctly show that the source is weak.

`P(Ia)` alone is not an abstention result. It answers which class is preferred,
not whether the observations contain enough source information to interpret the
answer.

## After selecting the spectral base

Add and measure one component at a time:

1. attention across the named ONIR regions;
2. an all-visit coadd context branch;
3. attention across visits using candidate rest-frame intervals;
4. a measured peak-date distribution evaluated within the redshift scan;
5. masked-spectrum pretraining of the encoder; and
6. consistency training across independent noise realizations.

These are useful comparisons, not requirements for starting the first NERSC run.
