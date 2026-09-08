# Calibration

Fit calibration after selecting and fixing the model weights. Use the reserved
calibration data and keep the test data for the final evaluation. STRIDER fits
three types of calibration:

1. **Class probability.** Binary models use a regularized affine transform of
   the Ia log odds (Platt scaling); multiclass models use one temperature. The
   binary artifact also records a cross-fitted maximum-F1 operating point and a
   high-purity point only when its one-sided 95% precision bound reaches the
   requested target.
2. **Redshift coverage.** The complete saved redshift marginal is converted to
   68% and 90% conformal highest-density sets. A set may contain disconnected
   intervals when the posterior is multimodal. Quantiles are conditioned on
   calibrated predicted class and visit bands 1--4, 5--16, and 17+, with a
   global fallback when a stratum has fewer than 200 calibration objects. The
   existing primary-basin interval remains a distinct local summary.
3. **Signal sufficiency.** The evidence score is mapped to the probability of a
   source signal under an equal mixture of source and matched noise-only inputs.
   High, medium, and low grade boundaries target noise-only false-positive
   rates of 0.1%, 1%, and 5%. This quantity is not redshift confidence.

For diagnostic checks, objects are split into two groups. Calibration fitted
on one group is evaluated on the other. Final parameters are then fitted using
all calibration objects. Every input parquet must carry `data_split`,
`data_view`, `checkpoint_epoch`, and `config_sha256`; the fitter refuses a test
split or a mismatched checkpoint/configuration.

## Commands

Once the checkpoint is frozen, save only the two views needed to fit the
calibration:

```bash
strider evaluate --config "$STRIDER_CONFIG" \
  --split calibration --views original no_source
strider fit-calibration --config "$STRIDER_CONFIG"
```

The second command writes these files in the configured run directory:

- `calibration.json`: portable parameters, provenance, operating points, and
  cross-fitted diagnostics;
- `calibration_summary.json`: compact diagnostic summary; and
- `calibration_predictions_original_calibrated.parquet`: raw outputs plus
  calibrated class probabilities, source probability/grade, and disconnected
  redshift sets.

After calibration is complete, run the declared final evaluation and export
the matching model package:

```bash
strider evaluate --config "$STRIDER_CONFIG" --split test
strider export-model --config "$STRIDER_CONFIG"
```

Export after calibration and final evaluation. Only results from a matching
test split are included. To replace an existing export while keeping a backup,
use `strider export-model --config "$STRIDER_CONFIG" --replace`.

Fit calibration separately for each model and class scheme. Calibration
parameters and grade thresholds cannot be transferred between them.

## Using calibration for inference

The calibrated columns do not overwrite raw network outputs. Deployment code
that retains the full `P(class, z)` tensor can use
`calibrate_joint_probability` to replace its class marginal while preserving
each raw `P(z | class)`. The conformal redshift sets are fitted
to the raw full redshift marginal, as recorded by `posterior_basis` in the
artifact. Model export includes a fitted `calibration.json` only when its config
digest and checkpoint epoch exactly match the exported weights.
