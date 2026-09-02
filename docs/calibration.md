# STRIDER post-training calibration

Calibration is a frozen-model stage. It does not train or alter STRIDER's
weights, choose a checkpoint, or read the Sundial test split. It fits three
products with deliberately separate meanings from the reserved calibration
blocks:

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
   source-bearing rather than matched no-source input under an explicit 50/50
   reference. High, medium, and low grade boundaries target blank false-positive
   rates of 0.1%, 1%, and 5%. This quantity is not redshift confidence.

Object-stable two-fold cross-fitting reports diagnostics without evaluating each
row with a calibrator fitted to that row. Final parameters are then fitted to
all reserved calibration rows. Every input parquet must carry `data_split`,
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

On NERSC, submit the CPU calibration job after the GPU evaluation succeeds:

```bash
eval_job=$(sbatch --parsable --time=06:00:00 \
  --export=ALL,STRIDER_CONFIG="$STRIDER_CONFIG",STRIDER_EVAL_SPLIT=calibration,STRIDER_EVAL_VIEWS="original no_source" \
  nersc/evaluate_test.sh)
cal_job=$(sbatch --parsable --dependency="afterok:${eval_job}" \
  --export=ALL,STRIDER_CONFIG="$STRIDER_CONFIG" \
  nersc/fit_calibration.sh)
test_job=$(sbatch --parsable --time=06:00:00 \
  --dependency="afterok:${cal_job}" \
  --export=ALL,STRIDER_CONFIG="$STRIDER_CONFIG",STRIDER_EVAL_SPLIT=test \
  nersc/evaluate_test.sh)
export_job=$(sbatch --parsable --dependency="afterok:${test_job}" \
  --export=ALL,STRIDER_CONFIG="$STRIDER_CONFIG" \
  nersc/export_model.sh)
echo "calibration: evaluation=$eval_job fit=$cal_job"
echo "frozen test=$test_job export=$export_job"
```

Do not export between calibration fitting and the frozen test evaluation. The
exporter includes metrics only from a matching test split and atomically replaces
an earlier package while retaining that package as a backup.

Fit one artifact independently for each binary, grouped-7, and 15-class model.
Do not transfer grade boundaries, temperatures, conformal quantiles, or
clearances between class schemes.

## Deployment contract

The calibrated columns do not overwrite raw network outputs. Deployment code
that retains the full `P(class, z)` tensor can use
`calibrate_joint_probability` to replace its class marginal while preserving
each raw `P(z | class)`. The conformal redshift sets in this version are fitted
to the raw full redshift marginal, as recorded by `posterior_basis` in the
artifact. Model export includes a fitted `calibration.json` only when its config
digest and checkpoint epoch exactly match the exported weights.
