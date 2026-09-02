# Next STRIDER steps

> **Historical planning record.** Tasks here are preserved for provenance and
> should not be read as the current release plan. See
> [`research_history.md`](research_history.md) and
> [`reproducibility.md`](reproducibility.md).

## Current decision — 2026-08-13

The current v3 checkpoint is the safety baseline, not the final architecture.
The full-spectrum shape-and-feature experiment has now shown that matching the
complete spectrum at every candidate redshift is useful alongside the named
ONIR features. The remaining binary comparisons ask one narrow question: does
relative spectroscopic amplitude evolution add useful information after
absolute object brightness is removed?

The runtime inputs remain observed wavelength, `FLAM`, observation date and a
valid-bin mask. `FLAMERR` is used outside the network to construct one robust
scale per visit; the wavelength-dependent error curve is not a model channel.
Truth redshift, truth phase, `SIM_FLAM` and simulation peak date are never
runtime inputs.

The candidate contains:

1. one mask-aware observer-frame encoder shared by all spectral routes;
2. one candidate-redshift matcher using both the whole spectral shape and a
   continuum-removed detail view;
3. a relative visit-amplitude trajectory with one object-wide gain removed;
4. candidate-rest-frame timing used only as an evolution-consistency check;
5. named ONIR regions retained as an interpretable auxiliary; and
6. a separate evidence-quality result trained with paired source-free inputs.

Continuum removal is subtractive. A mask-aware Gaussian smooth component is
subtracted and the valid bins are standardized. The width is configured in
km/s and converted to the current log-wavelength grid, so a debug grid cannot
silently apply a four-times broader filter. The present 12000 km/s value is a
prototype setting, close to the old v2 high-pass scale. A 150-object calibration
check found that narrower 3000--6000 km/s filters removed substantially more
source structure; 12000 and 18000 km/s remain the useful comparison before the
architecture is frozen. The unflattened view is always retained.

The whole and detail views share one encoder and one template matcher. Their
mixture may vary smoothly with trial redshift, because line detail is expected
to help most in the rest-frame optical while broad UV structure can matter
more at larger redshift. They produce one spectral result, not two independent
posterior branches.

The temporal term converts measured date intervals using
`delta_t_rest = delta_t_observed / (1 + z_trial)`. It can reject a spectrally
plausible but temporally inconsistent solution; it is not expected to measure
redshift by itself. A future measured photometric peak-date distribution may be
added as an explicit optional prior.

The final candidate advances only if absolute object gain leaves class-redshift
logits unchanged, correct spectra outperform shuffled or removed spectra,
source-free recovery remains near chance as visits increase, and performance
degrades with increasing noise. These gates are repeated after every capacity
or class-count change. The synthetic template library establishes internal
consistency only; OpenUniverse, SIRAH and eventually Roman-like external data
measure transfer.

For evaluation, report the posterior mode and median, 68% interval, secondary
peak mass, class probabilities and evidence grade. Apply any photo-z prior to
the saved posterior so its influence remains visible.

## Architecture decision now running

Four runs have distinct jobs and should not be conflated:

1. The full-spectrum shape-and-feature reference
   (`ia_binary_20k_dense_dual`) retains the full-spectrum context branch and
   scans both the unflattened spectral shape and continuum-removed features at
   every candidate redshift.
2. `ia_binary_20k_whole_detail` is an ablation. It removes the context branch
   and tests whether the scan plus relative brightness can stand alone. Its
   substantially weaker early classification is evidence that the context
   branch should remain; it is not a candidate for scale-up.
3. `ia_binary_20k_candidate` restores the full-spectrum shape-and-feature
   reference and adds the RMS amplitude of each background-scaled spectrum. It
   measures signal relative to the visit background and is retained as a
   conservative control, not as the definitive light-curve test.
4. `ia_binary_20k_relative_flux` reconstructs the flux-calibrated `FLAM`,
   integrates it over the measured observer-frame wavelength range, and divides
   the complete visit trajectory by one object-wide scale. It therefore removes
   absolute brightness while retaining the relative rise and decline. The
   stored FLAMERR scale is used only to undo visit-level preprocessing and
   cancels algebraically; it does not determine the resulting trajectory.

Keep relative amplitude only if it improves the validation selection result or
the physically important Ia and redshift metrics without weakening the
random-gain, source-free, noise-response, or visit-count controls. If both
amplitude variants are inactive or neutral, freeze the full-spectrum
shape-and-feature reference. If object-normalized `FLAM` is useful and safe,
freeze that version. Do not interpret a null result from the background-scaled
RMS arm as evidence that a Roman photometric light curve is uninformative, and
do not add another spectral route before this decision.

## Full-sample training after the architecture freeze

Create the final full-sample configuration only after the comparison above has
selected its winner. It should inherit that architecture unchanged, point to the
full prepared training and calibration data, and use the matching full ONIR
bank. The initial training recipe is:

```yaml
training:
  epochs: 30
  learning_rate: 0.0002
  learning_rate_schedule: cosine
  warmup_epochs: 2
  minimum_learning_rate_fraction: 0.05
  early_stopping_patience: 8
```

This retains the useful gradual convergence of v2 without repeating the
aggressive v3 schedule that rose to `1e-3` and became unstable in the earlier
full binary run. The schedule warms to `2e-4`, then decays to `1e-5`. Thirty
epochs is a ceiling rather than a claim that all thirty are required; validation
selection and early stopping choose the useful checkpoint.

NERSC time limits do not change this schedule. A continuation must use the same
resolved configuration and output directory and set `RESUME=1` when submitting
`nersc/train_model.sh`. The saved training state restores the model, optimizer,
learning-rate scheduler, completed epoch, random-number states, history and best
checkpoint information. Do not change the epoch ceiling or learning-rate
schedule halfway through this run. If the model is still improving at epoch 30,
define a separate, documented refinement stage initialized from the selected
checkpoint and using a lower learning rate.

The post-freeze order is:

1. run the development test on the final full configuration;
2. train and resume as necessary until early stopping or epoch 30;
3. run the route, source-free, visit-count, random-gain, noise-response and
   posterior-alias checks on the selected checkpoint;
4. calibrate class probabilities and apply class-conditional, visit-aware
   conformal calibration to redshift intervals using calibration data only;
5. freeze the checkpoint, resolved configuration, ONIR bank and calibration
   products as one model package; and
6. evaluate Sundial and external examples only after the package is frozen.

The fifteen-class model follows only after the binary evidence path passes these
checks. Re-run all safety and calibration checks after changing the class count;
do not inherit the binary clearances automatically.

## Immediate decision path

Do not run every diagnostic after every edit. Four checks govern the current
build:

1. clean source spectra show that the spectral route can learn the intended
   class and redshift structure;
2. generated and original noisy spectra measure useful performance;
3. matched source-free examples with the same dates remain broad and receive
   low evidence sufficiency;
4. changing visit dates or visit subsets does not create a confident result
   without corresponding spectral change.

The candidate-redshift temporal branch passes the structural and source-free
checks without using true redshift or truth-derived rest-frame phase. The
encoded ONIR branch and dense whole-spectrum scan now provide complementary
redshift evidence. Do not restore unrestricted date features or true phase to
recover score.

The current cluster comparison uses roughly 40,000 augmented training examples
and 10,000 validation examples from the 20k physical-object preparation. It is
large enough to choose between the two frozen binary candidates, but not to
quote final performance. The complete full-sample fit waits only for this
brightness decision and its controls; it does not wait for another architecture
search or a fifteen-class run.

Broader tests are attached to the component they investigate: named versus
random feature positions for ONIR, altered noise patterns for noise handling,
residual controls for coadding, and OpenUniverse plus multiple seeds before a
model is selected for scale-up.

## 1. Separate spectral evidence from temporal compatibility

Replace the direct addition of candidate-phase and spectral features.

- Produce phase-neutral class-redshift evidence from each visit.
- Compute temporal compatibility from changes between spectral embeddings and
  candidate rest-frame time differences.
- Expose both terms in evaluation output.
- Require the temporal term to become uninformative when spectral changes are
  replaced by no-source inputs.
- Retain matched no-source training with identical dates and masks.
- Use a weighted visit-count distribution rather than equal probability for
  every count; retain 1- and 2-visit examples without letting them dominate.

The first implementation is in `model/temporal.py`. It multiplies candidate
rest-frame time features by changes between consecutive spectral embeddings.
Identical embeddings produce zero time-dependent output, removing the earlier
direct phase-addition path. Cadence can still act through which spectra are
observed, masks, epoch count, and noise-induced embedding changes, so matched
source-free and changed-visit checks remain required. The first full local
result is recorded in `docs/spectral_evolution_results_2026_08_02.md`.

## 2. Improve controlled source-observation generation

The cadence-controlled models transfer poorly from generated observations to
original `FLAM`. Train with several source-bearing constructions while retaining
matched no-source controls for every construction:

1. clean signal plus source-free background noise;
2. clean signal plus a fresh reported-error draw;
3. an alternative detector or empirical noise model held out of training.

Encourage the source posterior to agree across independent noise constructions.
Do not train the class-redshift result to reproduce the original fixed noise
realization.

## 3. Scale the encoded named ONIR evidence

The first phase-neutral bank and raw-profile scan are implemented and locally
tested. They establish that clean profile shapes carry redshift information and
that exact named positions protect the high-redshift scan. They also establish
that raw cosine matching is not robust enough on the noisy pilot spectra.

The implemented encoded branch now:

- passes each visit through a shared compact spectral encoder;
- applies the tested candidate-redshift alignment and named-region gather;
- projects clean profile sets through the same representation;
- retains explicit support, wavelength validity, and overlap weights;
- uses three phase-neutral profiles per class-feature cell;
- exposes feature evidence for diagnosis; and
- keeps dates and phase out of this branch.

At scale, validate the generated and clean views together when selecting a
checkpoint. Retain the original `FLAM` view as a held-out reproduction test,
not a target that the model is trained to copy. See
`docs/encoded_onir_results_2026_08_02.md` for the local numbers.

Retain the clean mean, clean medoid, observed-FLAM, random-profile, and locally
randomized-position arms. Repeat the position comparison across several seeds at
full scale. The observed-FLAM bank remains a diagnostic rather than the default.

## 4. Retain cadence controls during training

The local controls show that the set of observer times predicts redshift even
when source flux is absent.  Training must vary this route while preserving
physically valid spectrum-time relationships.

- Randomly select visit subsets and phase spans independently of redshift.
- Vary peak-date estimates within their measured uncertainty.
- Treat evolution rate as a nuisance scale around \((1+z)\), not an exact rule.
- Where the clean simulator supports it, resample the time series onto changed
  observer cadences rather than merely changing time labels.
- Keep a held-out cadence construction that is never used in training.

Retain two reported diagnostics: spectral redshift evidence and temporal
evolution evidence. Their disagreement must lower evidence sufficiency or trigger
abstention; one branch should not silently overpower the other.

## 5. Strengthen evidence sufficiency and calibration

- Preserve `has_source`, continuous evidence-sufficiency target, and conditional
  class-redshift result as distinct fields.
- Train evidence sufficiency across several source-free and reported-error noise families.
- Hold one external noise construction out of training.
- Calibrate class probabilities, redshift intervals, and evidence sufficiency on a separate
  calibration subset after model selection.
- Report raw \(\Delta z\), population scatter, median absolute \(\Delta z\),
  outlier fraction, interval coverage, and N in every redshift group.

## 6. Scale classes only after the evidence path passes

The local sample is intentionally binary: normal Ia versus all other simulated
types.  Move to the fifteen-class output only after the ONIR and cadence tests
pass.  At full scale, split by source identity and simulation production, and
retain an alternative-simulator or empirical holdout where possible.

## 7. Later architecture comparisons

After the named-feature baseline is working:

- compare one standard and one larger encoder;
- add one attention block across named ONIR regions;
- add a coadded spectral context branch as a separate experiment;
- compare pairwise temporal compatibility with factorized temporal attention;
- test masked-spectrum pretraining only after the supervised scale-up model;
- compare feature count, window width, redshift-grid spacing, phase-bin count,
  hidden dimension, and visit aggregation;
- require no-source, residual, changed-cadence, and held-out-noise controls for
  every candidate.

The local coadded-only ONIR test did not improve source accuracy and increased
residual posterior peaks. A later coadded branch may supply a small object-level
context term beside the visit sequence, but it must not replace visit evidence,
use the error spectrum for weights, or be fused without its own residual and
no-source controls.
