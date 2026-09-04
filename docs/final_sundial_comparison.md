# Legacy and current Sundial comparison

> **Historical experiment record.** The route labels below match saved legacy
> and development artifacts. They are not versions of the public STRIDER tool.

This comparison has two deliberately separate questions.

1. **Primary noise response:** use every true Sundial Ia, the same objects,
   visits and native-bin noise draws in the legacy and current models, and
   compare redshift bias,
   redshift scatter, absolute redshift error and the fraction with
   `P(Ia) >= 0.9` as noise increases.
2. **Classification check:** use the complete mixed-class Sundial sample at
   the nominal reported uncertainty and measure Ia purity, completeness and
   probability calibration. This check is separate because purity is not
   defined on an Ia-only sample.

The existing paired diagnostic retains at most 32 visits in both models. That is
the fair comparison for the checkpoints already trained with that contract.
After the all-visit current model is trained, repeat only its final-test
evaluation and report it separately as the deployment model.

## NERSC setup

Start from the existing STRIDER environment. After a new login, restore the
five raw/prepared/output directory exports from `docs/nersc_runbook.md`, then:

```bash
cd /pscratch/sd/m/mdixon7/strider

export STRIDER_ROOT="$PWD"
export STRIDER_PYTHON=/global/homes/m/mdixon7/.conda/envs/strider/bin/python
export STRIDER_DATA_DIR=/pscratch/sd/m/mdixon7/strider/data_detector
export STRIDER_OUTPUT_DIR=/pscratch/sd/m/mdixon7/strider/runs_detector
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

: "${STRIDER_TRAIN_DIR:?restore STRIDER_TRAIN_DIR}"
: "${STRIDER_VALIDATION_DIR:?restore STRIDER_VALIDATION_DIR}"
: "${STRIDER_TEST_DIR:?restore STRIDER_TEST_DIR}"
```

## A. Primary matched-Ia noise comparison

Write every Sundial Ia to one manifest and eight deterministic shards:

```bash
comparison_dir="$STRIDER_DATA_DIR/paired_v2_v3"
ia_manifest="$comparison_dir/sundial_all_ia.csv"
mkdir -p "$comparison_dir"

"$STRIDER_PYTHON" scripts/build_paired_noise_manifest.py \
  --config configs/nersc/ia_binary_full_main.yaml \
  --split test \
  --all-ia \
  --shard-count 8 \
  --output "$ia_manifest"
```

Run v3. Ten hours is intentional: it has recently queued much faster than a
21-hour request, while eight shards keep each task manageable.

```bash
v3_ia_job=$(sbatch --parsable \
  --time=10:00:00 \
  --array=0-7 \
  --export=ALL,STRIDER_CONFIG="$PWD/configs/nersc/ia_binary_full_main.yaml",STRIDER_NOISE_MANIFEST="$ia_manifest",STRIDER_NOISE_TAG=sundial_all_ia,STRIDER_NOISE_SCALES="0.5 0.75 1.0 1.5 2.0 3.0",STRIDER_NOISE_REPEATS=2,STRIDER_NOISE_PAIRED_SEED=20260803 \
  nersc/paired_noise_check.sh)

echo "v3 all-Ia noise array: $v3_ia_job"
```

Run frozen v2 on the same manifest. This is a diagnostic of the legacy model,
including its known true-rest-frame phase shortcut.

```bash
v2_root=/pscratch/sd/m/mdixon7/strider
v2_output="$v2_root/runs/paired_v2_v3_noise/all_ia"

v2_ia_job=$(cd "$v2_root" && sbatch --parsable \
  --time=10:00:00 \
  --array=0-7 \
  --export=ALL,WORK_DIR="$v2_root",SUNDIAL_TRANSIENTS="$STRIDER_TEST_DIR",PAIRED_OBJECT_LIST="$ia_manifest",OUTPUT_DIR="$v2_output",COHORT=ia,NOISE_SCALES="0.5,0.75,1,1.5,2,3",NOISE_DRAWS=2,NOISE_SEED=20260803 \
  nersc/paired_v2_v3_noise.sh)

echo "v2 all-Ia noise array: $v2_ia_job"
```

After both arrays finish, create the primary comparison figure:

```bash
cd /pscratch/sd/m/mdixon7/strider

"$STRIDER_PYTHON" scripts/plot_paired_noise.py \
  --v2 "$v2_output" \
  --v3 "$STRIDER_OUTPUT_DIR/ia_binary_full_main" \
  --v3-tag sundial_all_ia \
  --output-dir "$STRIDER_OUTPUT_DIR/ia_binary_full_main/v2_v3_all_ia"
```

The main output is `true_ia_noise_v2_v3.png`. Its four columns show the
redshift bias, robust NMAD scatter, catastrophic-outlier fraction and median
`P(Ia)` in each true-redshift bin. The ordinary standard deviation remains in
the accompanying CSV as a tail-sensitive diagnostic, but it is not presented
as the typical redshift precision. `normal_noise_typical_errors.csv` gives the
nominal-noise values directly, including the median absolute error and the
fraction with `P(Ia) >= 0.9`.

## B. Mixed-class nominal check

This is the run that can measure real Ia purity and completeness. It uses one
nominal noise scale rather than repeating the complete stress sweep.

```bash
mixed_manifest="$comparison_dir/sundial_all_classes.csv"

"$STRIDER_PYTHON" scripts/build_paired_noise_manifest.py \
  --config configs/nersc/ia_binary_full_main.yaml \
  --split test \
  --all-classes \
  --shard-count 8 \
  --output "$mixed_manifest"

v3_mixed_job=$(sbatch --parsable \
  --time=10:00:00 \
  --array=0-7 \
  --export=ALL,STRIDER_CONFIG="$PWD/configs/nersc/ia_binary_full_main.yaml",STRIDER_NOISE_MANIFEST="$mixed_manifest",STRIDER_NOISE_TAG=sundial_mixed_nominal,STRIDER_NOISE_SCALES="1.0",STRIDER_NOISE_REPEATS=1,STRIDER_NOISE_PAIRED_SEED=20260803 \
  nersc/paired_noise_check.sh)

v2_mixed_output="$v2_root/runs/paired_v2_v3_noise/mixed_nominal"
v2_mixed_job=$(cd "$v2_root" && sbatch --parsable \
  --time=10:00:00 \
  --array=0-7 \
  --export=ALL,WORK_DIR="$v2_root",SUNDIAL_TRANSIENTS="$STRIDER_TEST_DIR",PAIRED_OBJECT_LIST="$mixed_manifest",OUTPUT_DIR="$v2_mixed_output",COHORT=all,NOISE_SCALES="1",NOISE_DRAWS=1,NOISE_SEED=20260803 \
  nersc/paired_v2_v3_noise.sh)

echo "v3 mixed nominal array: $v3_mixed_job"
echo "v2 mixed nominal array: $v2_mixed_job"
```

Then make the selection and calibration products:

```bash
"$STRIDER_PYTHON" scripts/plot_paired_noise.py \
  --v2 "$v2_mixed_output" \
  --v3 "$STRIDER_OUTPUT_DIR/ia_binary_full_main" \
  --v3-tag sundial_mixed_nominal \
  --output-dir "$STRIDER_OUTPUT_DIR/ia_binary_full_main/v2_v3_mixed_nominal"
```

Use `mixed_class_noise_summary.csv` for the numerical table,
`mixed_class_noise_v2_v3.png` for purity/completeness, and
`probability_calibration_v2_v3.png` for the reliability check.
