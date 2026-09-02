# All-visit coadd and phase-neutral ONIR test — 2026-08-03

## Question

Does coadding every recorded Sundial spectrum before the ONIR redshift scan improve
classification or raw redshift error? Does a bank constructed from similarly coadded
clean spectra work better than signatures pooled from individual phases?

## Setup

- 1,200 training, 300 validation and 500 held-out Sundial objects;
- binary normal-Ia versus other classification;
- every recorded visit retained: median 25, maximum 133;
- equal valid-bin coadd after the standard per-visit background-scale normalization;
- no `FLAMERR` input or error-based coadd weights;
- phase-neutral 15-region ONIR scan with 60 trial redshifts from 0.05 to 3.0;
- 20 training epochs and 771 trainable parameters;
- raw redshift errors use `delta_z = predicted_z - true_z`.

`SIM_FLAM` is used only to construct the two clean training-split banks and to
produce the clean diagnostic view. It is never passed to the fitted model or used
for held-out original observations.

Three otherwise matched arms were run:

1. scan each visit and average visit evidence;
2. coadd all held-out visits, scan once, and use profiles pooled from individual
   clean training visits;
3. coadd all held-out visits, scan once, and use profiles built by coadding each
   clean training object before extracting its ONIR windows.

Both banks deliberately use every recorded simulation phase for this test. The
pooled bank has 2,359 to more than 5,000 retained windows per class-feature cell.
The matched-coadd bank has 187 to 508 training objects per cell. Neither bank has
an unsupported cell.

## Overall held-out results

### Original SNANA observations

| arm | class accuracy | median absolute delta z | population scatter | outlier fraction |
|---|---:|---:|---:|---:|
| per-visit scan | 0.538 | 0.451 | 0.596 | 0.828 |
| coadd, pooled-phase bank | 0.544 | 0.373 | 0.609 | 0.712 |
| coadd, matched-coadd bank | 0.532 | 0.403 | 0.626 | 0.708 |

### Independently generated observations

| arm | class accuracy | median absolute delta z | population scatter | outlier fraction |
|---|---:|---:|---:|---:|
| per-visit scan | 0.552 | 0.399 | 0.563 | 0.738 |
| coadd, pooled-phase bank | 0.558 | 0.348 | 0.624 | 0.704 |
| coadd, matched-coadd bank | 0.564 | 0.350 | 0.630 | 0.666 |

### Clean spectra

| arm | class accuracy | median absolute delta z | population scatter | outlier fraction |
|---|---:|---:|---:|---:|
| per-visit scan | 0.726 | 0.028 | 0.397 | 0.372 |
| coadd, pooled-phase bank | 0.726 | 0.036 | 0.618 | 0.414 |
| coadd, matched-coadd bank | 0.720 | 0.026 | 0.628 | 0.346 |

On the independently generated view, the paired change in median absolute redshift
error relative to the per-visit scan was:

- pooled-phase coadd: -0.051, 95% interval [-0.089, -0.014];
- matched-bank coadd: -0.049, 95% interval [-0.090, -0.008].

The classification changes were +0.006 and +0.012, with both intervals spanning
zero. Coadding therefore improved the central redshift result modestly but did not
demonstrably improve classification in this local model.

## Generated observations by redshift

| redshift | per-visit median abs delta z | pooled coadd | matched coadd | per-visit / pooled / matched outliers |
|---|---:|---:|---:|---:|
| 0.00-0.75 | 0.347 | 0.065 | 0.066 | 0.58 / 0.49 / 0.49 |
| 0.75-1.25 | 0.314 | 0.239 | 0.321 | 0.66 / 0.62 / 0.61 |
| 1.25-1.75 | 0.092 | 0.119 | 0.086 | 0.46 / 0.54 / 0.47 |
| 1.75-2.25 | 0.446 | 0.372 | 0.311 | 0.99 / 0.87 / 0.79 |
| 2.25-3.10 | 1.003 | 0.853 | 0.850 | 1.00 / 1.00 / 0.97 |

The high-redshift direction improves, but the absolute result remains poor. This
experiment does not make a high-redshift precision claim plausible.

## Source-free and residual controls

| arm | no-source target lock | no-source largest joint probability | residual target lock | residual largest joint probability |
|---|---:|---:|---:|---:|
| per-visit scan | 0.060 | 0.029 | 0.064 | 0.031 |
| coadd, pooled-phase bank | 0.076 | 0.104 | 0.084 | 0.107 |
| coadd, matched-coadd bank | 0.086 | 0.114 | 0.082 | 0.120 |

The evidence-sufficiency output remains low on these inputs: no source-free object
exceeds 0.5, and 0.8% of residual objects do. Nevertheless, coadding increases the
largest joint class-redshift probability by about four times. The joint posterior
therefore cannot be treated as self-validating.

## Reading

1. An all-visit coadd is useful enough to retain as a separately visible context
   branch. It gives a modest, paired improvement in central redshift error.
2. It should not replace per-visit evidence. It does not improve classification,
   increases population scatter, and makes source-free posterior peaks stronger.
3. Constructing the bank from matched clean coadds is not clearly better than
   pooling clean individual phases. It reduces generated-view outliers relative to
   the pooled bank, but gives similar median error and stronger source-free peaks.
4. Coadding every phase can blur evolution. A fixed observer-frame span and a
   learned encoded coadd branch remain separate NERSC comparisons.
5. This is a small raw-profile ONIR model. The result supports testing a coadd
   context branch beside the encoded per-visit route; it does not settle the final
  trained STRIDER architecture.
