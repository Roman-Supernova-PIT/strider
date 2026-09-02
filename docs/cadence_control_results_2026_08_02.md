# Cadence-control experiments — 2026-08-02

> **Historical research record:** This page preserves an earlier experiment or
> decision. It does not define the current STRIDER tool. See
> [`architecture.md`](architecture.md) for the current design.

## Question

Can STRIDER retain useful spectral and temporal information without producing a
confident class or redshift result from the observation schedule alone?

The original local pilot associated visit count and observer-time span with
simulation redshift.  Its candidate-phase embedding was added directly to the
spectral representation, giving the network a path from dates to class-redshift
scores even when source flux was absent.

## Changes tested

- Every training source received a matched no-source counterpart with identical
  visit dates, wavelength coverage, and masks.
- Source examples retained the joint class-redshift and continuous adequacy
  targets.
- No-source examples received low adequacy, a broad redshift target, and a broad
  class target.
- Half of the no-source pairs used new source-free noise. Half used a fresh draw
  from the reported-error pattern.
- Visit counts were drawn independently of redshift.
- The visit-count evaluation used three repeatable random subsets rather than
  always choosing the earliest visit or fixed endpoints.
- A separate timing-only model received dates, visit count, and time span but no
  flux.

The original checkpoint and results were not overwritten.

## Timing-only result

On the 500 held-out objects, the timing-only diagnostic obtained:

| Quantity | Result |
|---|---:|
| Class accuracy | 0.856 |
| Median absolute Δz | 0.415 |
| Population scatter of Δz | 0.750 |
| Fraction with \(|\Delta z|>0.1\) | 0.842 |

The observation schedule strongly predicts the binary class label in this
simulation sample. It predicts redshift much less accurately than the full
model. This is a dataset association that a scientific model must not treat as
spectral evidence.

## Paired-training comparisons

Two full 1,200-object fits were run. Both used 300 validation objects and the
same 500 held-out test objects.

### Model A: training visits drawn from 4, 8, or 12

| Input | Class accuracy | Median absolute Δz | Population scatter | Outlier fraction |
|---|---:|---:|---:|---:|
| Generated source observation | 0.910 | 0.256 | 0.629 | 0.688 |
| Original `FLAM` | 0.806 | 0.516 | 0.950 | 0.834 |
| Clean `SIM_FLAM` | 0.928 | 0.183 | 0.649 | 0.648 |

Paired generated-source versus source-free no-source result:

| Quantity | Source | No source |
|---|---:|---:|
| Fraction within \(|\Delta z|=0.1\) | 0.312 | 0.320 |
| Median absolute Δz | 0.256 | 0.266 |
| Mean normalized redshift entropy | 0.820 | 0.970 |
| Mean 68% interval width | 1.22 | 1.75 |
| Mean adequacy | 0.482 | 0.084 |

The difference in within-threshold fractions was -0.008 with paired bootstrap
interval [-0.050, 0.034] and exact paired p=0.777. Source inputs produced much
narrower distributions, but no better point-estimate success at this threshold.

Residual and fresh reported-error no-source inputs were rejected strongly:

| No-source construction | Mean adequacy | Fraction above 0.5 | Median absolute Δz |
|---|---:|---:|---:|
| Source-free noise | 0.084 | 0.000 | 0.266 |
| `FLAM-SIM_FLAM` residual | 0.054 | 0.004 | 0.893 |
| Fresh reported-error draw | 0.058 | 0.004 | 0.817 |

### Model B: training visits drawn from 1, 2, 4, 8, or 12

This fit doubled the no-source redshift and class penalty weights.

| Input | Class accuracy | Median absolute Δz | Population scatter | Outlier fraction |
|---|---:|---:|---:|---:|
| Generated source observation | 0.826 | 0.231 | 0.597 | 0.694 |
| Original `FLAM` | 0.764 | 0.569 | 0.978 | 0.848 |
| Clean `SIM_FLAM` | 0.788 | 0.190 | 0.691 | 0.660 |

Paired generated-source versus source-free no-source result:

| Quantity | Source | No source |
|---|---:|---:|
| Fraction within \(|\Delta z|=0.1\) | 0.306 | 0.272 |
| Median absolute Δz | 0.231 | 0.333 |
| Mean normalized redshift entropy | 0.903 | 0.990 |
| Mean 68% interval width | 1.46 | 1.90 |
| Mean adequacy | 0.531 | 0.179 |

The paired within-threshold advantage was +0.034 with bootstrap interval
[-0.014, 0.082] and exact paired p=0.181. This is directional evidence for a
spectral redshift route, not a statistically decisive result.

## Repeated random visit subsets

The table reports means across three independently selected visit subsets for
Model B.

| Visits | Source class accuracy | Source median absolute Δz | Source outlier fraction | No-source adequacy | No-source redshift entropy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.740 | 0.562 | 0.892 | 0.009 | 0.983 |
| 2 | 0.809 | 0.441 | 0.862 | 0.058 | 0.986 |
| 4 | 0.841 | 0.373 | 0.828 | 0.054 | 0.988 |
| 8 | 0.855 | 0.320 | 0.779 | 0.092 | 0.989 |
| 12 | 0.847 | 0.286 | 0.745 | 0.181 | 0.990 |

Additional visits clearly improve source classification and redshift. The
no-source distributions remain nearly flat and adequacy remains below 0.2.
Their MAP values still retain a weak association with simulation redshift, so a
MAP-within-threshold statistic must not be interpreted without posterior width
and adequacy.

## What was learned

1. Matched no-source training successfully removes confident residual-only and
   reported-error-only results.
2. The timing-only class result proves that cadence is associated with class as
   well as redshift in the pilot sample.
3. Repeated visits contain useful information, but the current model mixes
   spectral evolution and schedule information too early.
4. Uniformly sampling all visit counts improves low-count robustness and the
   paired spectral redshift comparison, but reduces full-sequence class
   performance.
5. The poor transfer from generated observations to original `FLAM` shows that
   the current source-free noise generator is not yet an adequate model of the
   original observations.
6. Neither paired model is ready to select an ONIR implementation or support a
   precision claim. The controls are now suitable for grading the next model.

## Architecture decision

The next model should not add an unrestricted phase embedding to the spectral
representation. It should expose two evidence terms:

- phase-neutral spectral evidence from each visit;
- temporal compatibility calculated from changes in spectral representations
  and candidate rest-frame time differences.

The temporal term must require measured spectral change. Identical spectral
representations must give zero time-dependent output. This blocks the direct
date path but does not remove schedule effects through visit selection, masks,
epoch count, or noise-induced representation changes; those require continuing
controls. A coadded spectral branch can later supply an additional source-shape
term, but should be tested separately.
