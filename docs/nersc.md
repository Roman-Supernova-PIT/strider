# NERSC workflow

This is the short path for the active reference candidate. The batch scripts
never install packages or modify the environment during a job.

## Environment

Create the Python environment before submitting work, install this checkout,
then define project-specific paths:

```bash
export STRIDER_ROOT=/path/to/strider
export STRIDER_TRAIN_DIR=/path/to/training/simulations
export STRIDER_VALIDATION_DIR=/path/to/selection-and-calibration/simulations
export STRIDER_TEST_DIR=/path/to/held-out/test/simulations
export STRIDER_DATA_DIR=/path/to/persistent/prepared-data
export STRIDER_OUTPUT_DIR=/path/to/run-output
export STRIDER_PYTHON=/path/to/environment/bin/python
```

The validation and test variables are required by the shared launch guard even
when a particular command does not read those roles. Point them only at the
declared stores; do not substitute one split for another.

## Prepare and inspect

Choose the preparation configuration appropriate to the source simulation and
submit:

```bash
export STRIDER_CONFIG=configs/nersc/uncertainty_reference.yaml
sbatch nersc/prepare_data.sh
sbatch nersc/build_reference_bank.sh
```

The reference builder is allowed to read training labels and clean spectra. The
`uncertainty_reference.yaml` path writes the bank required by the active
candidate. Its output must be format v3 and should be retained with its metadata
and checksum.

## Active candidate

The candidate configuration preserves the existing two-epoch checkpoint digest:

```bash
export STRIDER_CONFIG=configs/nersc/reference_candidate_gate.yaml
sbatch nersc/train_model.sh
```

For a checkpoint that already exists, do not resubmit training. Evaluate only
the selection role:

```bash
export STRIDER_CONFIG=configs/nersc/reference_candidate_gate.yaml
bash nersc/evaluate_reference_candidate.sh
```

Run that command inside an allocation with the required GPU resources. It is a
small guard script rather than a separate Slurm job so it cannot silently fall
back to the inherited calibration split.

Do not run `nersc/evaluate_test.sh`, fit candidate calibration or open the test
role until matched selection results have been reviewed and the candidate is
frozen. The longer [`nersc_runbook.md`](nersc_runbook.md) is retained as an
experiment-era operational record; its old route names are not the public
starting point.
