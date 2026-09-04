# Coadd-first STRIDER design

## Decision being tested

The frozen STRIDER models remain the paper baseline. This pilot asks whether a
measurement-faithful inverse-variance coadd should replace visit-mean matching
inside the dense class--redshift scan. It is not permission to reopen a frozen
checkpoint or to select an architecture on the final Sundial test set.

The proposed information factorization is:

1. the inverse-variance coadd supplies the highest-S/N spectral shape for the
   primary dense class--redshift match;
2. individual visits retain named-region shape and spectral-evolution evidence;
3. object-normalized broadband evolution supplies relative brightness change;
4. the source-sufficiency head supplies an explicit abstention probability;
5. clean simulated flux is an auxiliary denoising target during training only.

This is one architecture with distinct measurement roles, not several
independent classifiers whose outputs are combined after selection.

## Coadd contract

For visit `i` and wavelength bin `lambda`, preparation provides

- scaled observed flux `f_i(lambda)`;
- log scaled standard deviation `log sigma_i(lambda)`;
- a measured-bin mask `m_i(lambda)`;
- the scalar `s_i` by which native FLAM and FLAMERR were divided.

The model first restores a common object-relative scale,

```text
f'_i = f_i s_i / s_object
sigma'_i = sigma_i s_i / s_object,
```

where `s_object` is the geometric mean of valid visit scales. The common factor
`s_object` only controls numerical range and cannot change inverse-variance
weights. The coadd and propagated error are

```text
F = sum_i[f'_i / sigma'_i^2] / sum_i[1 / sigma'_i^2]
Sigma = 1 / sqrt(sum_i[1 / sigma'_i^2]).
```

Only measured bins enter either sum. Bins whose propagated error exceeds three
times the object's median coadded error are excluded, and five per cent of the
log-wavelength interval is trimmed from each detector edge. These are declared
measurement-quality rules, not learned class- or redshift-dependent cuts.

Flux variance is propagated through observer-grid interpolation using squared
interpolation weights. Direct interpolation of FLAMERR or log FLAMERR is not
permitted. Inter-bin covariance created by interpolation is acknowledged; the
pilot uses per-bin precision only to combine independent visits, not to claim a
diagonal likelihood across wavelength.

The completed coadd is normalized once, after combination, for spectral-shape
encoding. Visits are never normalized independently before weighting.

## Runtime boundary

Runtime model inputs are limited to observed flux, reported/generated
measurement uncertainty, wavelength and visit masks, relative dates, and the
recorded visit preprocessing scale. Runtime inference must not receive class,
redshift, simulated phase, peak truth, or clean simulated flux.

`clean_flux_target` may be present in a training batch, but
`measurement_inputs()` excludes it. The auxiliary head reconstructs the
normalized clean coadd from the noisy coadd representation. Its prediction is
not an additional inference route and its loss does not define checkpoint
selection.

## Predeclared pilot comparison

The matched control is the selected all-visit, detail-only dense-scan recipe.
The pilot is a sequential ladder rather than one confounded comparison:

1. `coadd_only` changes only dense aggregation;
2. `coadd_denoise` adds the clean-coadd auxiliary target;
3. `coadd_first` adds object-normalized relative brightness evolution.

A rung is submitted only after the preceding rung passes its selection and
source-free gates. This makes each performance change attributable while
avoiding a broad hyperparameter sweep. Every rung uses the same object roles,
seed, initial checkpoint, optimizer schedule, class labels, redshift grid, and
validation views.

Architecture selection uses the independent selection role. Calibration and
the versioned final Sundial test remain unopened until the recipe is frozen.

## Required gates

The candidate is eligible for promotion only if all of the following hold:

1. **Correctness:** hand-calculated unequal-error examples match the model
   coadd and propagated error; padded/missing bins never contribute.
2. **Scale invariance:** changing arbitrary visit preprocessing scales while
   preserving physical FLAM and FLAMERR does not change the normalized coadd or
   dense class--redshift result.
3. **No source-free lock:** fresh-noise and no-source controls remain flat in
   redshift and do not acquire high source probability as visits accumulate.
4. **Joint inference:** class-balanced accuracy, Ia F1, Ia median absolute
   redshift error, and Ia outlier fraction are no worse than the matched control
   within predeclared tolerances.
5. **S/N behavior:** performance improves monotonically or remains stable with
   measured coadded S/N, and gains persist within class--redshift--S/N strata.
6. **Transfer:** improvements agree across original FLAM, fresh reported-error
   draws, controlled noise, and clean diagnostic views.
7. **Calibration:** held-out class reliability and 68/90 per cent redshift-set
   coverage do not degrade after the architecture is frozen and recalibrated.
8. **Cost:** the epoch time and memory increase are reported; a scientifically
   negligible gain does not justify a materially more expensive production
   model.

## Outcomes

- If the gates pass, the coadd-first recipe becomes the candidate for a new
  production training cycle and the public architecture description.
- If redshift improves but class separation degrades, keep the coadd as the
  redshift anchor and revise only how temporal evidence conditions class.
- If the source-free or transfer gates fail, reject the route regardless of
  in-distribution accuracy.
- If gains are negligible, retain the frozen architecture and report the pilot
  as evidence that visit-level matching was sufficient.
