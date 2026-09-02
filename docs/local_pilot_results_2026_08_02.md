# Local Sundial pilot results — 2026-08-02

## Outcome

The new STRIDER path runs from raw Sundial FITS files through native-bin
preparation, controlled observation generation, Metal training, held-out
inference, and diagnostic controls.  It is a working software and experimental
baseline.  It is not yet a replacement for the current STRIDER redshift model.

## Data and split

- Raw source: 90 complete HEAD/SPEC pairs, 10 FITS blocks.
- Training: blocks 1–7, 1,200 objects.
- Validation: block 8, 300 objects.
- Test: blocks 9–10, 500 objects.
- Each local split is balanced between normal Ia and all other simulated types.
- The sample is also balanced across five broad redshift groups, so it is not a
  population-rate sample.
- Prepared content: 61,967 visits and 12,315,730 native detector bins.
- Prepared size: about 114 MB.
- 57,931 catalogue rows across the ten blocks had no stored spectrum and were
  excluded before sampling.  None of the 2,000 selected objects is empty.

The prepared store keeps native wavelength bounds, original `FLAM`, clean
`SIM_FLAM`, `FLAMERR`, MJD, exposure, and object metadata as separately named
quantities.  New noise is drawn on the native bins and then resampled once.

## Local machine result

Training used an Apple M5 Pro with 64 GB unified memory through PyTorch Metal.
The final pilot has 19,043 trainable parameters.  The best checkpoint was epoch
19 of 20.

The worker benchmark mattered more than the model benchmark:

| Setting | Measured rate |
|---|---:|
| Model step, batch 24 | 550.4 objects/s |
| Direct HDF5 loading, 0 workers | 598.2 objects/s |
| HDF5 loading, 4 workers | 10.3 objects/s |
| HDF5 loading, 8 workers | 5.2 objects/s |
| HDF5 loading, 12 workers | 3.5 objects/s |

The local configuration therefore uses zero worker processes.  NERSC must be
benchmarked separately. The complete rerun is recorded in
`runs/sundial_local_pilot/benchmark_summary.json`.

## Held-out result

All values below use the 500 objects from FITS blocks 9–10.  Redshift values are
plain \(\Delta z=z_{\mathrm{pred}}-z_{\mathrm{simulation}}\).

| Input view | Class accuracy | Ia precision | Ia recall | Median absolute Δz | Population scatter of Δz | Outlier fraction, \(|\Delta z|>0.1\) |
|---|---:|---:|---:|---:|---:|---:|
| Original SNANA `FLAM` | 0.870 | 0.814 | 0.960 | 0.230 | 0.790 | 0.660 |
| Clean plus new source-free noise | 0.926 | 0.902 | 0.956 | 0.200 | 0.679 | 0.642 |
| Clean `SIM_FLAM` | 0.906 | 0.908 | 0.904 | 0.183 | 0.691 | 0.630 |
| Clean plus a fresh reported-error draw | 0.868 | 0.813 | 0.956 | 0.240 | 0.777 | 0.670 |

Binary classification is promising for such a small pilot.  Redshift recovery
is not yet adequate: the posterior is commonly broad or multi-peaked, and the
high-redshift group has the largest errors.  These probabilities have not been
calibrated, so they are diagnostic outputs only.

## Adequacy controls

The adequacy branch is separate from the conditional class-redshift result.  It
was trained against source-free no-source examples and no-source examples drawn
from reported SNANA errors.  Its target for simulated sources changes smoothly
around coadded clean S/N = 1.

| No-source input | Mean reported adequacy | Fraction above 0.5 | Median absolute Δz to simulation | Fraction within Δz=0.1 |
|---|---:|---:|---:|---:|
| Source-free noise | 0.165 | 0.000 | 0.267 | 0.334 |
| Original `FLAM-SIM_FLAM` residual | 0.249 | 0.070 | 0.274 | 0.342 |
| Fresh reported-error draw | 0.254 | 0.080 | 0.269 | 0.348 |

The residual and fresh reported-error controls do not recover redshift better
than the source-free no-source view.  This rebuilt pilot therefore does not
recover the earlier strong residual-only redshift result.  The similar
association with simulation redshift in all three no-source cases comes from
observer-time sampling, as the next control shows.

For original source observations, mean adequacy decreases with redshift:

| Redshift group | Mean adequacy |
|---|---:|
| 0.00–0.75 | 0.814 |
| 0.75–1.25 | 0.646 |
| 1.25–1.75 | 0.444 |
| 1.75–2.25 | 0.334 |
| 2.25–3.10 | 0.293 |

This is a continuous data-quality result, not a fixed redshift cut.

## Phase and observer-time controls

| Generated-input case | Class accuracy | Median absolute Δz |
|---|---:|---:|
| Normal observer times | 0.926 | 0.200 |
| Times assigned to the wrong spectra within each object | 0.886 | 0.229 |
| All times set to zero | 0.512 | 0.577 |

The within-object reassignment result shows that matching spectral evolution to
time contributes useful information.  The zero-time result shows that the
overall observer-time distribution contributes much more.

For source-free no-source inputs, 33.4% lie within \(\Delta z=0.1\) of the
simulation redshift with normal times, 34.2% after within-object reassignment,
and only 6.2% when all times are zero.  Therefore the no-source association is
carried by the set of visit times, not by spectrum-to-time pairing.

An otherwise identical phase-neutral model gives 0.720 generated-input class
accuracy and median absolute \(\Delta z=0.414\).  Its no-source fraction within
\(\Delta z=0.1\) is 5.6%.  The current pilot relies too strongly on phase and
observer-time sampling because its spectral redshift evidence is weak.

## Number of visits

| Visits | Generated class accuracy | Generated median absolute Δz | Original class accuracy | Original median absolute Δz |
|---:|---:|---:|---:|---:|
| 1 | 0.658 | 0.409 | 0.566 | 0.470 |
| 2 | 0.638 | 0.282 | 0.562 | 0.347 |
| 4 | 0.812 | 0.297 | 0.698 | 0.335 |
| 8 | 0.918 | 0.212 | 0.864 | 0.293 |
| 12 | 0.926 | 0.200 | 0.870 | 0.230 |

Repeated visits clearly improve classification.  Redshift improves overall but
not monotonically.  More visit times also strengthen the no-source redshift
association, so cadence augmentation is required before treating the gain as
pure spectral evidence accumulation.

## What the plots show

The three files in `runs/sundial_local_pilot/example_figures` contain four
held-out objects near each of \(z=0.75\), \(1.5\), and \(2.5\).  They compare
the original and clean spectra and show redshift distributions for original and
new-noise inputs.

At low redshift the clean source shape is visible on the noise scale and some
redshift distributions peak close to truth.  Around \(z=1.5\) the source shape
is weaker and secondary redshift peaks are common.  Near \(z=2.5\), the clean
curve is usually tiny compared with the original fluctuations and the redshift
result is broad or multi-peaked.  The local model is no longer implausibly
precise where the mean source is barely visible.

## Conclusions

1. The native-bin, controlled-noise data path works and is fast enough for local
   development.
2. `SIM_FLAM` is useful for generation and causal tests, not as a deployment
   input.
3. The rebuilt model does not show a special residual-noise redshift advantage.
4. The separate adequacy output can reject both source-free and SNANA-pattern
   no-source inputs when trained with the right target.
5. The present spectral scanner is too weak.  Candidate phase and the visit-time
   distribution carry too much of the redshift result.
6. The next scientific model needs named ONIR wavelength anchors and profile
   evidence before a larger attention backbone, coadded context, or fifteen-class
   run is justified.
