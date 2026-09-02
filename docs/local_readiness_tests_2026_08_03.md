# Local readiness tests — 2026-08-03

These tests decide what to run first on NERSC. They are architecture checks on
held-out Sundial objects, not performance forecasts for the full training set.
All redshift errors below are raw `predicted redshift - true redshift`.

## Why STRIDER 2 needs a different training contract

The completed native-bin noise run contains 15,342 Sundial objects, including
8,218 Ia. Fresh noise was drawn around `SIM_FLAM` on the native SNANA bins and
then resampled using the production error rule.

For Ia at `2.0 <= z < 2.5`, the frozen STRIDER 2 model gave:

| input | median absolute delta z | fraction above 0.1 | median P(Ia) |
|---|---:|---:|---:|
| original FLAM | 0.0060 | 0.101 | 1.000 |
| fresh noise at the reported scale | 0.0058 | 0.092 | 1.000 |
| fresh noise at three times the reported scale | 0.0065 | 0.132 | 1.000 |

At `z < 1`, tripling the noise changed the outlier fraction from 0.222 to
0.410. The inverted redshift dependence is not credible as a source-signal
measurement. Together with the earlier mismatched-variance and residual tests,
this confirms that the STRIDER noise changes are scientifically required.

## Mismatched reported-error pattern

The clean, phase-neutral ONIR scan was tested on 111 held-out Ia with
`1.35 <= z <= 2.5`. Each clean target spectrum was combined with either its own
fresh reported-error noise or an amplitude-matched noise realization from an Ia
at least 0.5 away in redshift.

| representation and input | target within 0.1 | noise source within 0.1 | median absolute delta z to target |
|---|---:|---:|---:|
| visits, target noise | 0.189 | 0.072 | 0.504 |
| visits, mismatched noise | 0.162 | 0.099 | 0.437 |
| coadd, target noise | 0.396 | 0.027 | 0.315 |
| coadd, mismatched noise | 0.225 | 0.072 | 0.417 |

The current representation does not follow the mismatched noise source. Coadding
improves the source result, but its absolute local accuracy remains too weak to
replace large-sample training.

## Coadd and phase stacks

Five hundred held-out Sundial objects were compared with the same untrained
clean ONIR scan and up to 32 visits.

| representation, original FLAM | class accuracy | median absolute delta z | high-z median absolute delta z |
|---|---:|---:|---:|
| individual visits | 0.538 | 0.463 | 0.800 |
| one all-visit coadd | 0.528 | 0.384 | 0.510 |
| two phase stacks | 0.512 | 0.442 | 0.614 |
| four phase stacks | 0.506 | 0.451 | 0.716 |

Phase-specific clean profiles improved some class results, but their original
FLAM median absolute delta z was 0.438 and the high-z value was 0.584. Peak-date
errors from 0.7 to 10 observer days did not explain the loss. Phase stacks are
therefore a later trained comparison, not the first NERSC representation.

Coadding also increased the mean largest joint probability on source-free noise
from about 0.03 to 0.15. A coadd branch must keep the no-source training and
evidence-sufficiency result; coadd confidence is not self-validating.

## ONIR profile interval

A clean phase-neutral bank built over the common `-20` to `+50` rest-day range
was compared with the current `-20` to `+80` bank. The shorter bank did not
improve the overall result. For original FLAM, its median absolute delta z was
0.518 for visits and 0.446 for the coadd, compared with 0.463 and 0.384 for the
current bank. Keep `-20` to `+80` for the first offline bank. This is a profile
construction interval, not an inference input cut.

## Timing remains a separate problem

The timing-only pilot reaches 0.856 binary class accuracy without receiving
flux. The first NERSC science fit must therefore be spectral-only. The temporal
branch may be trained only after a schedule-matched construction brings the
timing-only result to the appropriate class-frequency or permutation baseline.

## Code checks

- Full tests: 54 passed and 2 intentionally skipped.
- Shell syntax: every file in `nersc/` passes `bash -n`.
- A temporary train, evaluation and model-package export completed from source.
- Evaluation was repeated with runtime warnings treated as errors.
- Posterior entropy and information gain now remain finite when grid cells have
  exactly zero probability mass.
- The NERSC trainer now retains the useful v2 operational choices: A100
  bfloat16, pinned-memory transfers and gradient clipping at 1.0. The timing
  comparison uses the same loss terms as training.
- Visit accumulation is an explicit exponent. The first run uses a mean
  (`exponent=0`) rather than importing the value fitted to v2.

## What was retained from the NERSC-tested v2 path

STRIDER keeps AdamW, learning-rate warmup followed by cosine decay,
class-frequency weighting, gradient clipping, one-GPU shared jobs, held-out
prediction files and best-checkpoint selection. It adds native-bin streaming,
independent selection/calibration/test roles, repeatable noise generation,
atomic continuation with random states, and a self-contained model directory.

Do not carry over v2's warm start, EMA weights or multi-GPU memory workaround
for the first run. The new model and data contract are different, and the
streaming loader removes the host-memory reason for booking unused GPUs. EMA can
be tested later against the ordinary best checkpoint.

## NERSC decision

Run these in order:

1. inspect the new simulation blocks and class counts;
2. prepare and benchmark a bounded sample;
3. prepare the complete native-bin data and build the clean ONIR bank;
4. train `configs/nersc/spectral.yaml`;
5. evaluate original, generated, clean, residual and no-source views;
6. run the timing-only comparison on the full split;
7. consider a coadd reference and temporal model only after the baseline controls
   pass.

The first fit scans the complete declared range on a 500-point grid uniform in
`log(1+z)`. It does not promise useful precision everywhere. Report performance,
feature support and evidence sufficiency as functions of redshift, signal to
noise and visit count. High-redshift claims return only where those measurements
support them.

Do not assume a `z < 2` validity region. Frozen-model source-removal tests show
the first leak sensitivity near `z=1.4` and dominant noise-pattern control near
`z=1.8`. Use `z < 1.4` as the clean v2 reference range, then report finer bins
through the transition. Keep the complete redshift range in training and
evaluation so the model learns and demonstrates when evidence becomes weak. Do
not add a hard inference cut, and do not weight training labels with `SIM_FLAM`
signal to noise: that simulation-only quantity must not become a hidden
redshift cut.
