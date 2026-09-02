# Encoded ONIR local results — 2026-08-02

## Scope

This is an architecture test on the local binary Sundial subset, not a final
performance measurement. It uses 1,200 training, 300 validation, and 500
held-out objects. The model has 3,127 trainable parameters.

The tested branch encodes every observer-frame visit once, gathers nine
fractional token positions around each of 15 named rest-frame regions for every
trial redshift, and compares those tokens with three clean, phase-neutral
profiles per class-feature cell. Dates, phase, and the reported error spectrum
are absent from this branch.

## Architecture result

The encoded matcher is a useful improvement over direct raw-profile matching.
On generated noisy spectra, it changed class accuracy from 0.558 to 0.636 and
median absolute raw redshift error from 0.668 to 0.337. On original `FLAM`, it
changed class accuracy from 0.526 to 0.632 and median absolute raw redshift error
from 0.749 to 0.378.

It also preserved the clean redshift mechanism: median absolute raw redshift
error is 0.020 on clean spectra. This matters more than a small local ranking:
the same compact branch can read physical spectral structure in clean inputs
and retain part of it under two noisy observation constructions.

## Held-out comparison

| Input | Class accuracy | Median \(|\Delta z|\) | Population scatter of \(\Delta z\) | Mean adequacy |
|---|---:|---:|---:|---:|
| Generated noisy spectra | 0.636 | 0.337 | 0.630 | 0.505 |
| Original `FLAM` | 0.632 | 0.378 | 0.693 | 0.420 |
| Clean spectra | 0.650 | 0.020 | 0.383 | 0.582 |
| Reported-error signal plus new noise | 0.622 | 0.363 | 0.667 | 0.417 |

Source-free controls did not recreate the earlier precise-redshift behaviour:

| Control | Fraction within \(|\Delta z|<0.1\) of simulation redshift | Mean adequacy | Fraction adequacy above 0.5 |
|---|---:|---:|---:|
| Generated no-source | 0.078 | 0.127 | 0.000 |
| `FLAM-SIM_FLAM` residual | 0.082 | 0.074 | 0.002 |
| Reported-error no-source | 0.086 | 0.075 | 0.000 |

In matched source/source-free pairs with identical visit dates, the fraction
within \(|\Delta z|<0.1\) was 0.302 with source and 0.078 without source. The
paired gap was +0.224 with a bootstrap 95% interval of [0.180, 0.268]. This is
direct evidence that the local model uses spectral source information rather
than dates alone.

## Visit-count result

Generated-input median \(|\Delta z|\) improved from 0.428 with one visit to
0.333 with two and 0.301 with four. It then stayed near 0.30 at eight and twelve
visits. The architecture benefits from repeated observations, but its current
uniform evidence average does not extract much additional information after
about four visits. This motivates a later tested aggregation improvement; it
does not justify adding unrestricted date features.

## Longer training result

A second run allowed up to 40 epochs and selected epoch 30 using generated-view
validation loss alone. Relative to epoch 20:

- generated accuracy improved from 0.636 to 0.662;
- generated median \(|\Delta z|\) improved from 0.337 to 0.326;
- original accuracy improved from 0.632 to 0.646;
- original median \(|\Delta z|\) improved from 0.378 to 0.372;
- clean accuracy fell from 0.650 to 0.608;
- clean population scatter increased from 0.383 to 0.428, while clean median
  \(|\Delta z|\) remained approximately 0.021.

The longer checkpoint is therefore not uniformly better. The training loop now
supports declared validation views and a weighted checkpoint score. The NERSC
run should validate on generated and clean source-bearing views together, while
retaining matched source-free examples in the generated view. A partial local
run confirmed that both losses are computed and recorded separately; it was
stopped because the 1,200-object sample is too small for architecture ranking.

## Decision

The direction is accepted for scale-up:

1. retain the compact shared encoder;
2. retain the named candidate-redshift gather and explicit feature support;
3. retain clean phase-neutral profile initialization with several profiles per
   class-feature cell;
4. retain generated, original, clean, residual, and matched no-source tests;
5. use multi-view checkpoint selection at scale;
6. do not claim high-redshift performance from this local run.

The local model is still weak above the lower redshift groups and has only two
classes. The next meaningful performance test needs the larger training set,
the fifteen-class target, several noise constructions, multiple random seeds,
and a held-out simulation or detector-noise construction.
