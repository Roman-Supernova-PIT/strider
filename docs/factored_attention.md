# Factored spectral and temporal evidence

> **Historical research record:** This page preserves an earlier experiment or
> decision. It does not define the current STRIDER tool. See
> [`architecture.md`](architecture.md) for the current design.

## Why this comparison exists

The first encoded ONIR model deliberately averages named regions and visits.
That is useful as a simple reference, but it cannot learn that different
regions matter for different classes or that particular spectral changes are
consistent with a candidate rest-frame interval.

The factored model changes only that evidence combination. It retains the clean
ONIR bank, log-wavelength redshift scan, native-bin noise generation, joint
class-redshift posterior and separate evidence-sufficiency output.

## Data path

```text
one encoder for every visit
    -> named ONIR regions at every trial redshift
       -> class queries combine spectral shape across regions
       -> consecutive encoded visits measure spectral change
          -> observed gap / (1 + trial redshift)
          -> attention combines changes across visits and regions
    -> one object-level joint class-redshift posterior
```

The model estimates one redshift for the complete transient, not one redshift
per visit. Dates cannot score a class or redshift by themselves. They only
change how measured spectral evolution is interpreted. Identical visit spectra
therefore produce an exactly zero temporal logit.

`FLAMERR` remains important for native-bin noise generation, valid-bin masks,
coadds and evaluation controls. Its wavelength-dependent pattern is not passed
to the classifier as another spectral channel in this comparison.

## Controlled example

`strider temporal-example` creates evolving feature sequences with random
rest-frame gaps and random starting phases from -25 to +25 days. It stretches
the observer intervals by `1 + z` and contains no SNANA noise, cadence or
population balance.

Three independent runs used 1,400 training and 500 test objects each:

| dates | class accuracy | median absolute delta z |
|---|---:|---:|
| correct | 100.0% | 0.171 |
| reversed | 42.4-58.4% | 1.029-1.371 |
| reassigned | 99.8-100.0% | 0.686-0.857 |

Identical visits give a maximum temporal logit of exactly zero. This test also
found and removed a learned normalization offset that previously broke that
rule after training.

When every object starts at the same phase, the correct-date redshift error
falls from one grid step to zero. The visit-gap route can learn time dilation
without supplied phase, but unknown starting phase is a real nuisance variable.

## Learning phase without using it as an input

A second controlled check trained a small network to predict each visit's phase
from its spectral features. True simulated phase was a training target only.
On 600 random-start test objects it reached a median phase error of 0.17 days.
Combining those predicted phases with the observer intervals gave median
absolute delta z of 0.011; reassigning dates between objects raised it to 0.850.

This is an easy synthetic example, not a Roman result. It does show that phase
prediction and visit-gap evidence are complementary. A STRIDER phase head may
be trained with simulation truth, provided truth phase is never passed into the
runtime model and the predicted phase uncertainty is retained.

This proves that the branch can learn the intended relationship. It does not
measure performance on Roman simulations.

## What carries forward from STRIDER 2

STRIDER 2 used a CNN stem followed by alternating spectral attention within
each visit and temporal attention at each wavelength patch. Its feature head
also attended across visits for each named ONIR region. Those are useful ideas,
but the temporal blocks did not receive the elapsed time between visits. The
input tokens had already been conditioned on simulated rest-frame phase, and a
second input channel contained adjacent normalized-spectrum differences.

STRIDER keeps the shared spectral encoder, named-region representation and
separate spectral and temporal attention. It changes the temporal contract:

- simulated phase is a training target or bank-building value, never an input;
- measured observer-frame gaps are evaluated at every trial redshift;
- spectral change and elapsed time meet in one explicit branch;
- one visit or identical visits provide no temporal evidence; and
- spectral, temporal and evidence-sufficiency outputs remain separately visible.

This retains the useful factorization without relying on truth redshift to set
phase, an unscaled difference channel, or visit attention with no time axis.

## Cadence stress test

A larger controlled run used 20,000 training sequences and 5,000 test
sequences. Training rest-frame gaps were 4--18 days; test gaps were widened to
2--26 days. Correct dates gave 100% class accuracy and median absolute redshift
error of one 0.171 grid step. Reassigning dates between objects increased the
error to 0.857, and reversing the dates increased it to 1.20. Performance
improved from two to three visits and then stabilized.

The branch therefore learns the intended time-dilation relation rather than a
date-only lookup. Its one-grid-step bias outside the training cadence range is
also a warning: training must vary visit gaps broadly, and Roman evaluation
must report performance by visit count and time span.

A harder binary run used 50,000 training and 10,000 test sequences, one normal
Ia family and fourteen distinct non-Ia families. It added object-to-object
spectral variation, stronger feature noise, 2--5 training visits, 31 redshift
cells and test gaps extending beyond training. Correct dates gave 100% binary
accuracy and median absolute redshift error 0.16. Reassigned dates increased
the error to 0.80 and reversed dates to 0.96. With one visit, classification
remained possible from shape but redshift error was 1.20; two and five visits
reduced it to 0.24 and 0.16 respectively.

This is the intended division of work. Spectral shape separates Ia from the
contaminants, the clean ONIR scan will provide the principal wavelength-based
redshift location in Roman data, and temporal evidence checks whether the
measured evolution agrees with each trial redshift.

## What the NERSC comparison measures

Run `spectral_full_bank.yaml` and `factored_20k.yaml` from scratch. They share
the same 20,000 training objects, validation roles, full clean bank, redshift
grid, optimizer and training length.

Evaluation records two additional decompositions:

- redshift with the true class fixed, which tests ONIR redshift matching; and
- class with the true redshift fixed, which tests class separation.

Every source-bearing view also writes one row per class with N, precision,
recall, F1, median delta z, delta-z scatter, median absolute delta z, outlier
fraction, 68% coverage and mean evidence sufficiency.

The factored model advances only if:

1. source-bearing class and redshift results improve or remain comparable;
2. Ia precision improves with recall, rather than predicting Ia for everything;
3. noise-only and observed-minus-clean inputs remain broad and low-sufficiency;
4. correct visit dates outperform reversed and reassigned dates; and
5. confidence remains stable as visit count changes.

If it passes, the next step is a full-data training run. Coadding, uncertain
peak date, masked-spectrum pretraining and larger encoders remain separate later
comparisons rather than being bundled into this one.
