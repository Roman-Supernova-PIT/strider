# Roman spectral-reference candidate

> **Selection rationale.** This document records the question and first-gate
> comparison that led to the current candidate. For the implemented runtime,
> corrected edge treatment and present status, use
> [`architecture.md`](architecture.md) and
> [`research_workflow.md`](research_workflow.md).

## Question being tested

Can STRIDER obtain a stronger, easier-to-explain class and redshift estimate by
matching the final measured coadd and its spectral evolution to multiple clean
training references for each physical transient class?

This is a selection-only candidate. It does not replace the frozen production
model unless it passes the predefined comparison.

## Information boundary

Training simulations provide class, redshift, and phase only while the
reference bank is constructed. Those labels place clean simulated spectra on a
common rest-wavelength and broad-phase grid. The deployed model receives only:

- measured observer-frame flux;
- the reported uncertainty of each measurement;
- the wavelength and visit masks; and
- observation dates relative to the first retained visit.

It never receives the measured object's true class, redshift, simulated clean
flux, or simulated phase.

## Model path

1. Reverse the numerical visit scaling and form an inverse-variance coadd with
   the reported uncertainty belonging to each spectrum.
2. Compare the full coadd and its continuum-removed view with multiple clean
   training references over every candidate redshift.
3. Retain a small set of visits that jointly covers the observed time span and
   favors the best measured S/N within each part of that span.
4. Compare those visits with broad phase-indexed references. Integrate over
   possible starting phases; do not provide the simulated phase of the object.
5. Match the 15 physical classes first, then combine their evidence exactly
   into the requested 15-class, grouped 7-class, or Ia-versus-other output.
6. Add the coadd match and spectral-evolution consistency into one joint
   class--redshift surface. A separate calibrated output reports whether the
   measurements contain enough reliable signal to interpret that surface.

The reference shapes may adjust slightly during training. A drift penalty keeps
them close to their clean training initialization and preserves their physical
meaning.

## Why this differs from earlier routes

- It uses the full Roman prism interval rather than making 15 named wavelength
  regions the model bottleneck.
- It keeps multiple fine-class and phase references rather than one learned
  template per reported output class.
- The coadd is the primary redshift anchor; visits test spectral evolution.
- S/N determines measurement weight and trust. It is not a class label.
- Named spectral regions remain useful for explaining a result, but no longer
  decide what information the matcher is allowed to inspect.

## First gate

Build references from `train`, train on `train`, and compare only on
`selection`. Calibration and test remain unopened. The first seed contains two
matched reference models:

- **Direct reference:** mask-aware correlation between measurements and the
  physical references, with equal averaging over supported sequence visits.
- **Learned reference:** the same bank, data, coadd, phase grid, and loss, plus
  a small shared mask-aware CNN for local spectral features and continuous-time
  attention over supported visits.

This separates the value of the physical reference design from the value of
learned feature extraction and visit weighting. Compare both with the
coadd-first v3 control. Promote only a model that produces a material and
repeatable improvement, especially for true Ia above redshift 1.5 and within
matched measured-S/N ranges, without degrading source-free controls.
