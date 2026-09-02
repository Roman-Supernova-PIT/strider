# NERSC runbook

> **Historical experiment record.** This longer runbook preserves active paper
> and NERSC provenance, including earlier route names. It does not define the
> current reference candidate. Start with [`../README.md`](../README.md),
> [`architecture.md`](architecture.md) and [`nersc.md`](nersc.md).

## 1. Prepare the environment once

Use a tested Python environment containing the versions in `pyproject.toml`,
then install this checkout in editable mode. Record the exact package versions;
batch jobs do not install packages.

```bash
python -m pip install -e .
python -m pytest -q
mkdir -p logs
```

Set these paths before every submission:

```bash
export STRIDER_ROOT=/path/to/strider
export STRIDER_TRAIN_DIR=/path/to/PIP_JP_HOURGLASS2_FLATDNDZ_TRANSIENTS
export STRIDER_VALIDATION_DIR=/path/to/PIP_DJ_AUG03_SUNDIAL_FLAT_TRANSIENTS
export STRIDER_TEST_DIR=/path/to/PIP_JP_HOURGLASS2_FIXED_TRANSIENTS
export STRIDER_DATA_DIR=$SCRATCH/strider/data
export STRIDER_OUTPUT_DIR=$SCRATCH/strider/runs
export STRIDER_PYTHON=/path/to/the/tested/python
export STRIDER_CONFIG=configs/nersc/spectral.yaml
```

## 2. Inspect before preparing

The supplied full config uses all ten blocks of the original flat-redshift
production for training, blocks 1--8 of the new August flat-redshift production
for model selection, blocks 9--10 of that production for calibration, and all
ten blocks of the separate Sundial production for the final test. Confirm those
files and block numbers before preparing data.

```bash
$STRIDER_PYTHON -m strider.cli inspect --config $STRIDER_CONFIG
```

Check the actual block count and rare-class spectra. Preparation fails if the
training role lacks any requested class. A test role without a rare class is not
a measurement of that class and must be reported as such.

If the independent validation production is still being transferred, the
training source can be checked on its own without assigning training objects to
a validation role:

```bash
sbatch nersc/check_training_source.sh
```

This reads at most 5,000 training objects and writes a separate prepared store.
It does not read the validation or final-test productions and is not used to
select a model.

## 3. Run the small end-to-end cluster test

The first complete run uses 300 training, 150 selection, 150 calibration and
150 final-test objects. It checks the data roles, clean ONIR construction and a
GPU forward/backward pass without waiting for the larger storage measurement.

```bash
export STRIDER_CONFIG=configs/nersc/dev.yaml
prepare_job=$(sbatch --parsable --qos=debug --time=00:30:00 nersc/prepare_data.sh)
bank_job=$(sbatch --parsable --qos=debug --time=00:30:00 \
  --dependency=afterok:$prepare_job nersc/build_onir_bank.sh)
sbatch --qos=debug --time=00:30:00 \
  --dependency=afterok:$bank_job nersc/development_test.sh
```

## 4. Measure storage and loading

Prepare a bounded sample before writing the complete dataset. This uses the same
scientific fields and model as the production config, but limits each data role.

```bash
export STRIDER_CONFIG=configs/nersc/storage_test.yaml
prepare_job=$(sbatch --parsable nersc/prepare_data.sh)
sbatch --dependency=afterok:$prepare_job nersc/build_onir_bank.sh
```

Compare worker counts on this sample and inspect GPU waiting time. If the single
HDF5 file limits loading, change only the physical grouping to multiple files;
keep the scientific fields and split assignments unchanged.

## 5. Run the first spectral pipeline

NERSC currently allows 30 minutes in `debug` and up to 48 hours in the GPU
`shared` quality of service. Shared jobs are charged for their fraction of the
node, so the one-GPU jobs below request one quarter of a four-GPU node. Recheck
the policy page when submitting because queue limits can change.

The local timing-only comparison still predicts class from the observation
schedule. The base NERSC configuration therefore contains no temporal branch.
See `local_readiness_tests_2026_08_03.md` for the deciding tests.

Prepared stores must include the native wavelength endpoints used by the
simulation templates. Use a new `STRIDER_DATA_DIR` for a different support
policy; do not overwrite another prepared store or ONIR bank. The simulation
directories are read only. Preparation writes only under `STRIDER_DATA_DIR`.

```bash
export STRIDER_CONFIG=configs/nersc/spectral.yaml
prepare_job=$(sbatch --parsable nersc/prepare_data.sh)
bank_job=$(sbatch --parsable --dependency=afterok:$prepare_job nersc/build_onir_bank.sh)
test_job=$(sbatch --parsable --dependency=afterok:$bank_job nersc/development_test.sh)
train_job=$(sbatch --parsable --dependency=afterok:$test_job nersc/train_model.sh)
sbatch --dependency=afterok:$train_job nersc/evaluate_model.sh
```

`development_test.sh` uses the `debug` queue for the test suite and a bounded
one-GPU forward/backward and worker check. `train_model.sh` uses `shared`, first
measures data and model throughput, then trains on one GPU. If a job ends after
a completed epoch, continue with:

```bash
RESUME=1 sbatch nersc/train_model.sh
```

Continuation requires the same fully resolved configuration and restores model,
optimizer, early-stopping and Python/NumPy/Torch random-generator states.

### Five-epoch training checks

After the 20k prepared data and ONIR bank exist, a five-epoch run can reuse both:

```bash
export STRIDER_CONFIG=configs/nersc/spectral_20k_5epoch.yaml
sbatch --time=02:00:00 nersc/train_model.sh
```

That run still sees all 20,000 training objects and normally needs more than the
30-minute debug limit. For a debug-queue training check, build the separate small
bank once and train on bounded runtime subsets:

```bash
export STRIDER_CONFIG=configs/nersc/spectral_debug.yaml
bank_job=$(sbatch --parsable nersc/build_onir_bank.sh)
sbatch --qos=debug --time=00:30:00 \
  --dependency=afterok:$bank_job nersc/train_model.sh
```

The debug result checks that training starts, loss decreases and files are
written correctly. It is not used to compare scientific performance.

Once the five-epoch path has passed, the measured runtime also permits a
20-epoch fit in a separate debug job:

```bash
export STRIDER_CONFIG=configs/nersc/spectral_debug_20epoch.yaml
sbatch --qos=debug --time=00:30:00 nersc/train_model.sh
```

This starts a new fit and leaves the five-epoch checkpoint unchanged.

After the spectral-only comparison, test whether the information-strength
output rejects all three supported no-source constructions:

```bash
export STRIDER_CONFIG=configs/nersc/spectral_debug_noise.yaml
sbatch --qos=debug --time=00:30:00 nersc/train_model.sh
```

This run shares each noise realization between a source and no-source partner.
Half use the controlled background and half draw from the reported uncertainty.
The original observation residual remains unseen and provides an independent
check. Keep this change only if both trained no-source views and the unseen
residual remain broad and receive low information strength.

If blank information strength still rises with visit count, repeat the same fit
without giving observation count or span to that component:

```bash
export STRIDER_CONFIG=configs/nersc/spectral_debug_no_schedule.yaml
sbatch --qos=debug --time=00:30:00 nersc/train_model.sh
```

The class-redshift model is unchanged. This comparison asks only whether
information strength can be estimated from the measured spectra themselves.

Compare both fitted models on the same larger calibration sample:

```bash
export STRIDER_EVAL_OBJECTS=1600

export STRIDER_CONFIG=configs/nersc/spectral_debug_noise.yaml
schedule_eval=$(sbatch --parsable --qos=debug --time=00:30:00 \
  nersc/evaluate_source_blank.sh)

export STRIDER_CONFIG=configs/nersc/spectral_debug_no_schedule.yaml
spectral_eval=$(sbatch --parsable --qos=debug --time=00:30:00 \
  nersc/evaluate_source_blank.sh)

echo "With schedule summaries: $schedule_eval"
echo "Spectra only:            $spectral_eval"
```

This option changes only the number of calibration objects read after the
checkpoint has been verified. It does not alter the fitted model or overwrite
the original 300-object result.

The first larger fit uses 50,000 objects sampled across all ten training blocks:

```bash
export STRIDER_CONFIG=configs/nersc/spectral_50k.yaml

prepare_job=$(sbatch --parsable nersc/prepare_data.sh)

test_config=configs/nersc/spectral_50k_test.yaml
test_train=$(sbatch --parsable --qos=debug --time=00:30:00 \
  --dependency=afterok:$prepare_job \
  --export=ALL,STRIDER_CONFIG="$test_config" \
  nersc/train_model.sh)
test_eval=$(sbatch --parsable --qos=debug --time=00:30:00 \
  --dependency=afterok:$test_train \
  --export=ALL,STRIDER_CONFIG="$test_config" \
  nersc/evaluate_model.sh)

train_job=$(sbatch --parsable --dependency=afterok:$test_eval \
  nersc/train_model.sh)
evaluate_job=$(sbatch --parsable --dependency=afterok:$train_job \
  nersc/evaluate_model.sh)

echo "Preparation: $prepare_job"
echo "Two-epoch test: $test_train"
echo "Test evaluation: $test_eval"
echo "Training:    $train_job"
echo "Evaluation:  $evaluate_job"
```

This is a twelve-epoch comparison run. A later job can continue the selected
checkpoint without restarting its optimizer or learning-rate schedule.

## 6. First cluster checks

Before committing the complete production to this physical storage layout:

1. prepare the bounded storage-and-loader dataset above;
2. test 1, 2, 4, 8 and 16 workers on disjoint object ranges;
3. retain persistent workers for multi-epoch measurements;
4. record objects/s, GPU waiting time, open file count and memory; and
5. split the native-bin arrays across several files if one HDF5 file limits the GPU.

The scientific schema remains fixed: native wavelength bounds, original flux,
reported error, clean simulated flux, object data and visit data. Only the
physical grouping of those arrays should change after the measurement.

The training job records `nvidia-smi dmon` samples in its output directory. Use
them with the Python benchmark table to decide worker count, batch size and
walltime. The 24-hour request in the script is a starting ceiling, not a measured
requirement; replace it after measuring seconds per epoch and allowing a 1.5
margin.

## 7. Required outputs

Every successful run writes:

- `config.resolved.yaml` and `environment.json`;
- `training_state.pt` for exact continuation;
- `best_model.pt` and `training_history.json`;
- `training_progress.png`, made from the history with `strider plot-training`;
- prediction Parquet files with split identity and full class probabilities;
- per-class CSV tables with precision, recall, F1 and direct-redshift metrics;
- redshift probability arrays when enabled;
- per-object evidence maps with the joint posterior and supported ONIR regions;
- visit-by-visit evidence GIFs from `nersc/plot_evidence.sh`;
- matched source/no-source and visit-count controls; and
- `model_package/` with weights, grids, bank, preprocessing, metrics and
  `SHA256SUMS`.

The first 15-class run chooses a checkpoint using only the selection role.
Calibration is fitted later on the calibration role; the test role remains
unread until the model and calibration procedure are fixed.

Check a frozen model across lower and higher noise without retraining it:

```bash
export STRIDER_CONFIG=configs/nersc/ia_binary_20k.yaml
sbatch nersc/noise_check.sh
```

Repeat with `ia_binary_20k_noise_range.yaml` to compare the same scales. The
reported blank redshift match must remain flat while source performance changes.

## 8. Normalization comparison

The base spectral model centres and scales each visit before the encoder. Its
evidence-sufficiency component still sees background-scaled flux, so absolute
signal information is not discarded from the complete output.

The first matched comparison retains background-scaled flux in the spectral
encoder as well:

```bash
export STRIDER_CONFIG=configs/nersc/spectral_scaled.yaml
train_job=$(sbatch --parsable nersc/train_model.sh)
sbatch --dependency=afterok:$train_job nersc/evaluate_model.sh
```

Both arms reuse the same prepared data and clean ONIR bank. Select between them
using the selection role and require all of the following:

1. source-bearing redshift and class results improve or remain comparable;
2. source-free evidence sufficiency remains low;
3. source-free class-redshift probabilities remain broad;
4. performance does not arise from brightness alone; and
5. the result remains stable across generated and reported-error views.

Do not enable temporal evolution while making this comparison.

Use `z < 1.4` to check that a new model retains the clearly source-driven v2
result. Report finer bins from `z=1.4` through `z=2.2`, where the frozen model
changes from source-driven to noise-pattern-driven behavior. Do not assign v3 a
fixed validity boundary in advance: source-removal, mismatched-noise and
evidence-sufficiency results determine it. The inference scan remains
continuous across the configured range.

Run the timing-only comparison on the complete split before enabling temporal
evidence. A temporal result is quoted only against the selected normalization
arm. The supplied temporal configuration currently extends the shape-normalized
base; change its parent only if the background-scaled arm wins the declared
comparison.

```bash
export STRIDER_CONFIG=configs/nersc/temporal.yaml
$STRIDER_PYTHON -m strider.cli timing-baseline --config $STRIDER_CONFIG
```

If timing alone remains above its class-frequency or permutation baseline, do
not submit the temporal model. First change the simulated schedule construction
and repeat the timing-only comparison. Any later temporal model uses the same
prepared data, ONIR bank and evaluation roles as the spectral baseline.

## 9. Factored shape and evolution comparison

The full clean ONIR bank is shared by both arms. The first arm keeps the current
region average. The second learns attention over named regions and compares
encoded visit changes at each trial redshift. Data, optimizer, redshift grid and
training length are otherwise unchanged.

```bash
base=configs/nersc/spectral_full_bank.yaml
factored=configs/nersc/factored_20k.yaml

base_train=$(sbatch --parsable --export=ALL,STRIDER_CONFIG=$base nersc/train_model.sh)
factored_train=$(sbatch --parsable --export=ALL,STRIDER_CONFIG=$factored nersc/train_model.sh)

sbatch --dependency=afterok:$base_train \
  --export=ALL,STRIDER_CONFIG=$base nersc/evaluate_model.sh
sbatch --dependency=afterok:$factored_train \
  --export=ALL,STRIDER_CONFIG=$factored nersc/evaluate_model.sh
```

The 20,000-object comparison answers whether the added structure is useful. It
does not choose final model size or replace the later full-data training. Read
the per-class tables as well as the averages: high Ia recall with low precision
means the model is predicting Ia too often.

Keep the factored branch only if it improves source-bearing class and redshift
results without making the no-source posterior narrow. Correct visit dates must
also beat reversed and reassigned dates; otherwise the temporal component is
not measuring the intended evolution.
