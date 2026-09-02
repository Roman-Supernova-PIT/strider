# Separated spectral and temporal evidence — local result

> **Historical research record:** This page preserves an earlier experiment or
> decision. It does not define the current STRIDER tool. See
> [`architecture.md`](architecture.md) for the current design.

## Purpose

The earlier pilot added candidate-phase features directly to every spectral
representation. That allowed observation dates to influence class and
redshift even when the spectrum contained no source.

The new development branch makes two explicit quantities:

- phase-neutral spectral class-redshift logits;
- temporal logits formed from changes between consecutive spectral
  representations and candidate rest-frame time intervals.

If the measured spectral representations do not change, the time-dependent
part of the temporal logits is exactly zero. A learnable scale begins at zero,
so training starts from the spectral-only result. Real source-free inputs still
contain noise changes, and the observing schedule controls which spectra exist;
the architecture therefore reduces one route rather than proving complete
independence from dates.

## Structural checks

- Dates cannot generate temporal evidence when spectral representations are
  identical.
- One-visit objects receive zero temporal evidence.
- The phase-neutral spectral logits are unchanged when dates are replaced.
- At initialisation, the complete result is exactly the spectral-only result.
- The complete test suite passes: 21 tests.

## Full local comparison

Both rows use the same 1,200 training, 300 validation, and 500 held-out objects,
the same visit-count choices, paired source/source-free training, and the same
seed. This is a one-seed engineering comparison, not a final model-selection
result.

| Input | Direct phase addition: class accuracy | Separated evidence: class accuracy | Direct phase addition: median abs. Δz | Separated evidence: median abs. Δz |
|---|---:|---:|---:|---:|
| Generated source | 0.826 | 0.678 | 0.231 | 0.352 |
| Original FLAM | 0.764 | 0.742 | 0.569 | 0.588 |
| Clean SIM_FLAM | 0.788 | 0.654 | 0.190 | 0.356 |

| Source-free control | Direct phase addition: near simulation z | Separated evidence: near simulation z |
|---|---:|---:|
| Generated source-free noise | 0.272 | 0.054 |
| FLAM − SIM_FLAM residual | 0.094 | 0.024 |

For paired generated-source and source-free examples, the fraction within
|Δz| ≤ 0.1 was 0.224 versus 0.054. The paired difference was +0.170 with a
bootstrap 95% interval of [0.130, 0.212]. Source-free posterior entropy was
0.992 and mean adequacy was 0.143.

Replacing or reversing the visit dates changed generated-source class accuracy
by at most 0.8 percentage points and left the source-free near-z rate between
5.2% and 5.6%. The learned temporal scale was 0.127 at epoch 20.

The saved prediction tables now record centred logit strength for each branch.
On generated source inputs the median temporal-to-spectral strength ratio was
0.027; on original FLAM it was 0.086. The residual ratio was larger at 0.245,
but both terms were weak in absolute amplitude and the residual posterior
remained broad. This measurement will make later temporal changes directly
auditable instead of inferring their effect from the combined result.

## Reading

The architectural restriction succeeds at its narrow purpose: identical
spectral representations cannot acquire redshift structure through direct date
addition. On this held-out sample, source-free inputs also remain broad under
date changes. That empirical result is not a general proof that dates cannot
enter through noise changes, masks, visit selection, or epoch count. The lost
source accuracy shows that the present full-spectrum encoder does not recover
all of the useful spectral information that direct phase addition had obscured.

This does not justify returning to direct phase addition. It identifies the
next target: strengthen the phase-neutral spectral route using an encoded ONIR
scan with interpolated trial-redshift positions, explicit wavelength support,
and multiple clean phase-neutral profiles. Only after that route works should
the local-plus-context stem be compared.

## Follow-up implementation correction

The first temporal normalization layer had a learnable offset. With zero
spectral change that offset was constant across redshift, so it could not carry
date-dependent redshift structure, but it could contribute a constant class
term. The layer now has no learned scale or offset. This makes the intended
contract exact after training: zero spectral change produces zero temporal
class and redshift evidence. The first checkpoint above is retained as the
recorded development comparison; the corrected implementation uses a separate
run directory.

## Tests deferred deliberately

- Multiple training seeds are required when choosing between final models, not
  for establishing that the structural restriction works.
- OpenUniverse is required before adopting a final encoder, not for this local
  implementation check.
- Coadding is a separate later component and keeps its own source-free and
  residual controls.
- Broadening and altered-noise studies are run after the encoded ONIR branch is
  functional, because they cannot improve the current weak spectral route.
