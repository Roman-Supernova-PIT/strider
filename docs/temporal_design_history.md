# Spectral and temporal design

> **Historical design record.** This document explains an earlier transition;
> it is not the public description of the current tool. See
> [`architecture.md`](architecture.md).

## Decision

STRIDER is the safer base, but its current factored ONIR model is a controlled
comparison rather than the final architecture. It fixes the information routing
that made STRIDER 2 hard to interpret. It does not yet contain all of STRIDER 2's
useful spectral capacity.

The next model should keep STRIDER's inputs, clean ONIR scan, noise controls
and evidence-sufficiency output. It should recover the useful CNN and factored
attention ideas from STRIDER 2 without copying its true-phase input or raw
pixel-difference channel.

## What STRIDER 2 did

STRIDER 2 encoded each visit with a CNN and alternated two attention operations:

1. spectral attention across wavelength patches within a visit;
2. temporal attention across visits at the same wavelength patch.

It also had a global spectral-shape head, local named-feature heads and an ONIR
redshift scan. This was a strong way to learn both broad shape and local
features.

Its temporal interpretation had three important limitations:

- true rest-frame phase modified the backbone tokens, feature queries, feature
  weights and phase-binned ONIR signatures;
- the temporal attention received a visit mask but not the actual time gaps;
- the second input channel was the difference between adjacent normalized raw
  spectra, so visit order and amplified noise entered before the redshift scan.

True phase therefore supplied much of the physical time coordinate. Removing it
without replacing that coordinate would not be a fair test of the attention
design.

## What STRIDER does now

The current factored ONIR model:

- encodes observed-frame flux and masks one visit at a time;
- gathers 15 named regions for every trial redshift;
- compares those regions with clean, phase-neutral ONIR profiles;
- uses class queries to combine spectral regions;
- measures changes between encoded visits;
- converts observer intervals to candidate rest intervals with
  `delta_time / (1 + trial_redshift)`;
- keeps spectral and temporal logits separate;
- makes identical visits produce exactly zero temporal logit; and
- reports evidence sufficiency separately from the class-redshift posterior.

This is easier to test and deploy. Its current encoder is much smaller than the
STRIDER 2 backbone and it sees only the named ONIR regions after encoding. It
may therefore lose useful broad spectral shape or weak features outside the
catalogue.

## Direct comparison

| Part | STRIDER 2 | Current STRIDER | Keep or change |
|---|---|---|---|
| Flux encoder | CNN stem | small mask-aware CNN | test a larger mask-aware CNN |
| Spectral context | transformer attention over all patches | named-region attention | restore a full-spectrum spectral block |
| Temporal context | attention at fixed wavelength patches | attention over encoded ONIR changes | retain both only if each adds validation value |
| Time coordinate | supplied true rest phase | observer gap divided by trial `1 + z` | keep the STRIDER route |
| Absolute phase | supplied to the model | absent | predict a phase distribution from spectra |
| Change input | raw adjacent-spectrum difference | difference of encoded visits | keep the STRIDER route |
| ONIR profiles | noisy FLAM-derived, phase-conditioned | clean SIM_FLAM-derived, phase-neutral | keep clean profiles; test phase bins separately |
| Noise handling | one fixed FLAM realization and per-object scaling | original and independently generated views with controls | keep STRIDER views and controls |
| Weak evidence | joint softmax still chooses an answer | separate evidence-sufficiency output | keep the separate output |

## Phase should be predicted, not supplied

Simulation truth is appropriate as a training label. It is not appropriate as
an inference input.

Add a class-conditioned phase head that predicts a distribution over rest-frame
phase for each visit. Start with 5-day bins from -20 to +50 days. Train it from
simulation phase, and score:

- phase error and interval coverage by class, redshift and signal-to-noise;
- consistency of phase ordering across visits;
- clean, independently noised and original FLAM views; and
- broad or low-confidence phase output for no-source inputs.

Do not feed its single best phase back into the model. During the redshift scan,
compare the predicted phase distributions with the candidate phases implied by
trial redshift and an optional observer-frame peak-date estimate. Marginalize
over peak-date uncertainty. The public result remains `p(class, redshift)`;
per-visit phase distributions are useful diagnostic output.

The observer-gap branch remains available when no peak-date estimate exists.
An external light-curve estimate can improve the phase calculation, but the
model must accept a missing estimate because one light-curve fitter will not be
reliable for every transient class.

## Controlled results

The learned temporal example contains no SNANA cadence, class balance, FLAMERR
or population-redshift relation.

With random starting phases, correct visit dates gave 100% class accuracy and
median absolute delta z of 0.171 in three runs. Reassigning dates raised the
redshift error to 0.686-0.857; reversing them raised it to 1.029-1.371. Identical
visits produced zero temporal evidence.

A separate phase-prediction check used true phase only as a target. It achieved
0.17-day median phase error in the simple synthetic data. Predicted phase plus
correct observer intervals gave median absolute delta z of 0.011, while
reassigned dates gave 0.850. These results establish capacity only; Roman
simulation tests still decide whether the routes are useful with realistic
noise and incomplete wavelength coverage.

## Coadded redshift proposal

An all-visit observed-frame coadd is a useful separate experiment. Build a
matched training reference by coadding each training time series in the same
way, then scan those phase-averaged profiles to obtain a broad proposal over
class and redshift. This can recover weak, persistent wavelength structure that
is hard to measure in individual visits.

The coadd and per-visit branches use the same photons, so their outputs are not
independent probabilities. The first implementation should use the coadd to
suggest likely redshift ranges while retaining the complete ONIR scan as a
fallback. It must report how often the true redshift lies in the proposed range.
Only a later jointly trained and calibrated model should combine their logits.

Compare three inputs without changing the rest of the model: one all-visit
coadd, a background-scaled coadd, and a small set of broad phase-group coadds.
Source-free coadds must return broad proposals. If phase grouping helps, the
groups must be formed from measured observer dates or a peak-date distribution,
never from truth-derived rest phase at inference.

## Matched full-spectrum comparison

Configuration: `configs/nersc/factored_context_20k.yaml`

This model restores the clearest missing capacity from STRIDER 2: a mask-aware
CNN followed by attention across the complete observed spectrum. It feeds one
class-context score per object into the factored ONIR model. It does not receive
dates, phase, truth redshift or per-bin uncertainty.

The context score is constant across the redshift grid. It can improve class
selection, and therefore the joint result, but it cannot create redshift
structure within a fixed class. ONIR remains the redshift anchor and the named
region temporal branch remains the measured evolution route. This makes the
comparison much closer to STRIDER 2's capacity without restoring its unsafe
inputs.

## Implementation order

1. Finish the matched 20,000-object spectral ONIR and factored ONIR runs.
2. Run the matched full-spectrum context comparison. Implemented in
   `configs/nersc/factored_context_20k.yaml`.
3. Add the phase head as an auxiliary output. Do not let it alter class-redshift
   logits yet. Implemented in `configs/nersc/factored_phase_20k.yaml`.
4. Confirm that phase is learned on clean data and degrades sensibly with noise.
5. Test the all-visit coadded redshift proposal as a separate arm.
6. Add phase compatibility inside the trial-redshift scan and compare it with
   the gap-only branch.
7. Add full-spectrum temporal attention only if the named-region evolution
   branch leaves a measured class or alias-rejection gap.
8. Keep only components that improve the independent validation data while the
   no-source, residual, date-reassignment and visit-count checks remain sound.

This gives three interpretable comparisons rather than one bundled rebuild:

- spectral ONIR;
- spectral ONIR plus safe temporal evidence;
- factored ONIR plus full-spectrum class context;
- the selected spectral model plus the tested phase route.
