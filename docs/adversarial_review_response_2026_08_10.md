# STRIDER candidate review response

## Decision

The STRIDER candidate remains a conditional go. The design is oracle-free and its blank
controls are encouraging, but a model result counts only when spectral input
adds clear value beyond cadence and coverage metadata.

The acceptance criteria are guardrails, not targets to tune against. We will
compare effect sizes and their uncertainty rather than optimize for a small,
arbitrary numerical margin.

## Reference model

The first full binary reference uses:

- normal Ia versus all other transients;
- observer-frame spectra and relative visit dates;
- candidate-dependent rest-frame gaps, `dt / (1 + z)`;
- phase-neutral ONIR profiles and factored spectral-shape evidence;
- the temporal route initialized at zero;
- no full-spectrum context branch;
- source-independent generated detector noise;
- no reported-FLAMERR noise mixture or noise-scale augmentation;
- a visit evidence exponent of zero;
- paired blank examples and an evidence-support output.

This is `configs/nersc/ia_binary_full_reference.yaml`. More expressive routes
are experiments until they improve spectral performance without improving any
blank or metadata-only control.

## What must be true

1. **Spectra add information.** The trained model must materially outperform
   timing-only and cadence-plus-coverage baselines on the same held-out objects.
2. **Blank input stays blank.** Source-free redshift lock must remain consistent
   with its permutation chance rate, with no meaningful redshift correlation or
   improvement as visits are added.
3. **Noise produces honest degradation.** Redshift error should grow and support
   should fall as injected noise grows. Precision must not remain implausibly
   fixed once the source disappears.
4. **The selection function is visible.** Results must report which objects were
   excluded by the complete-template policy, by class and redshift.
5. **The atlas cannot encode class through missing bins.** Every active mean,
   medoid, and prototype mask must equal the declared feature geometry. Sparse
   class-feature cells remain unsupported rather than acquiring a distinctive
   partial mask.
6. **Temporal evidence earns its place.** It remains only if measured spectral
   evolution improves held-out performance and the gain disappears when times
   are reassigned.

The natural precision limits vary strongly with redshift and signal strength,
so all principal metrics will also be reported by redshift and source support.

## Repairs completed after the review

- Route and evidence-growth checks now use one shared class-redshift-stratified
  object subset.
- Evidence-growth tuning uses the selection split, never calibration.
- Blank reports include lock, permutation chance, redshift correlation, and
  source-versus-blank support AUC.
- The timing-only model now reads visit count, span, dispersion, and gap
  statistics rather than nearly constant date features.
- A gradient-boosted metadata baseline measures cadence alone and cadence plus
  detector support without reading flux.
- The obsolete FLAMERR clipping option and the misleading noise-family leakage
  probe were removed.
- ONIR construction requires complete feature-window support, and both bank
  construction and loading reject active profile masks that differ from the
  declared geometry.
- Evaluation summaries carry the template-support selection function.
- A small result-bundling script excludes checkpoints and prepared data so NERSC
  JSON, CSV, and plots can be downloaded and audited locally.

## Run order

1. Rebuild the reference ONIR bank with the new geometry check.
2. Run the metadata-only and timing-only baselines on full selection data.
3. Train the binary reference model with resumable checkpoints.
4. Evaluate source, blank, residual, reported-error, noise-response, visit-count,
   route, and time-reassignment views.
5. Compare spectra with both metadata baselines on the same objects.
6. Add one component at a time: context, broader noise augmentation, temporal
   scale, and evidence growth.
7. Expand to 15 classes and repeat every shortcut test before using Sundial.

Sundial remains the final test set and is not used to choose architecture,
augmentation, calibration, or thresholds.
