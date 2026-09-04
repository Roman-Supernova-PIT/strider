# STRIDER model design

> **Historical experiment record.** This document predates the current
> architecture-aware package boundary. See [`architecture.md`](architecture.md)
> and [`data_and_models.md`](data_and_models.md) for the current contract.

## Aim

STRIDER should classify a transient and estimate redshift from a series of
Roman prism spectra. More visits should improve a result only when they add
repeatable source information. A weak or source-free series must remain broad
and carry a low evidence-strength score.

## What STRIDER 2 established

STRIDER 2 showed that three ideas are valuable:

- a CNN can turn each prism spectrum into useful local features;
- alternating wavelength and visit attention can combine a spectral series;
- an ONIR scan can connect named rest-frame features to redshift.

The trained model cannot be used as the starting point for a corrected run.
Its inputs included true rest-frame phase, its raw difference channel amplified
noise, its profile bank was made from noisy spectra, and one object-wide
normalisation removed the indication that a source was weak. At very low source
signal, repeated simulator structure could produce an unjustifiably precise
answer. Those choices are entangled in the saved weights.

## First classification target

The first deployable model separates normal SNe Ia from every other simulated
transient. The `other` class includes 91bg, Iax, all core-collapse subtypes and
the rarer classes. This keeps the first science question aligned with the Ia
sample needed for cosmology. The 15-class output is restored only after the
binary model passes the source-free and visit-count checks.

## The model to train

The model has four measured components.

1. **Full-spectrum context.** A mask-aware CNN is followed by alternating
   wavelength attention and visit attention. The visit attention has no visit
   position or date input, so reordering a complete set of visits cannot change
   this class-context result.
2. **ONIR redshift scan.** Fifteen named rest-frame regions are evaluated at
   every trial redshift on a grid uniform in `log(1 + z)`. Clean simulated flux
   builds the reference profiles. This branch anchors the redshift axis.
3. **Encoded spectral change.** Differences are taken between learned regional
   features, not raw flux bins. Observer intervals become candidate rest-frame
   intervals through `delta_time / (1 + trial_redshift)`. This is the explicit
   evolution route and remains separately measurable.
4. **Evidence strength.** A separate output reports whether the series contains
   enough measured source information to use the conditional class-redshift
   posterior. Dates, visit count and total span do not enter this output in the
   initial model.

The joint class-redshift map is the sum of ONIR, full-shape, class-context and
encoded-change logits. Every component is retained in the evaluation output so
its contribution can be checked.

## Phase

Simulated rest-frame phase is a useful training label and an invalid inference
input. A comparison model predicts a class- and redshift-conditioned phase
distribution for each visit from -20 to +50 rest-frame days.

The local comparison did not improve class or redshift performance, so the
first 20,000-object model leaves this head off. The phase loss may update the
shared spectral representation in a later comparison, but its logits must not
change the class-redshift map until that comparison shows a gain. A future
model may compare predicted phase distributions with an external light-curve
peak-date distribution. It must integrate over peak-date uncertainty rather
than use one fixed phase.

## Flux and uncertainty

The network receives one observed flux series. Clean flux, original SNANA flux,
independently generated noise and source-free inputs are separate views of an
object, not simultaneous channels.

- Clean simulated flux is used to build ONIR profiles and to measure the best
  performance available from the representation.
- Main training noise is drawn on native bins before resampling and has no
  source-dependent variance term.
- Original SNANA flux and noise drawn from reported `FLAMERR` are realism tests.
- The wavelength-dependent `FLAMERR` pattern never enters the network.

Flux is divided by one visit-level noise scale before the model. The current
simulation preparation estimates that scale from the lower part of reported
uncertainties. This preserves a measurable signal-to-background scale without
passing the wavelength pattern. A Roman deployment should replace it with a
background/read-noise estimate from the exposure or blank sky.

The full-spectrum and ONIR shape encoders then centre and scale each visit.
This prevents the visit-level noise scale from becoming a class or redshift
feature. The separate evidence-strength output retains the background-scaled
amplitude needed to identify weak measurements.

The initial accumulation is normalised across visits. A later comparison may
add one bounded quality weight per visit. Per-bin inverse-variance weighting is
not used until it is shown to leave source-free predictions broad for every
visit count.

## Required checks

The model advances only when all of these hold on data not used for fitting:

- source-free and observed-minus-clean inputs have low evidence strength and a
  broad class-redshift map;
- adding source-free visits does not increase target-redshift recovery;
- changing the reported noise pattern does not move the prediction to the
  pattern source;
- generated-noise performance is stable across independently drawn noise;
- reversing the spectra while keeping dates fixed weakens genuine temporal
  compatibility;
- reordering spectra, masks and dates together leaves order-independent routes
  unchanged;
- phase error and interval coverage are reported by class, redshift and source
  signal strength;
- class, redshift and evidence-strength metrics are reported separately in
  redshift and source-signal bins.

High-redshift precision is expected to become worse when the simulator route is
removed. That is a correct result when the measured source signal is weak.
Claims should follow evidence strength rather than a fixed redshift cut, while
detector feature-coverage limits remain separate hard boundaries.

## Run order

1. Run all unit tests and the local MPS benchmark.
2. Prepare a fresh bounded local sample with fixed detector coverage.
3. Build its clean ONIR bank and train the short spectrotemporal model.
4. Confirm that source-free redshift agreement does not improve with visit
   count. The local binary model passed this check from 4 to 32 visits.
5. Train the same binary model on 20,000 NERSC training objects.
6. Apply the source-free, reported-error, visit-count and time controls.
7. Train the complete binary flat-redshift sample only if those checks pass.
8. Select on the independent flat
   validation production, calibrate once, then open Sundial for the final test.

The working NERSC configuration is
`configs/nersc/ia_binary_20k.yaml`. The full NERSC configuration is
`configs/nersc/ia_binary_full.yaml`. The bounded Mac configuration is
`configs/experiments/local_spectrotemporal_noise_current.yaml`; after the
prepared 20,000-object data are copied from NERSC, use
`configs/experiments/local_ia_binary_20k.yaml`.
