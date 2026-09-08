# Training and evaluation

The public repository provides the scientific pipeline, portable configurations
and tests. Commands below run directly through `strider`; choose computing
resources appropriate to the data and record them with the run. A supported
checkpoint and a locked release environment are still pending.

## Installation

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The `strider example` command in the main README checks the
installation using artificial time series. It needs no simulation files or
trained model.

## Data preparation

Run from the repository root and set the locations for your authorized simulation
products and local outputs:

```bash
export STRIDER_TRAIN_DIR=/path/to/flat-training
export STRIDER_VALIDATION_DIR=/path/to/flat-validation
export STRIDER_TEST_DIR=/path/to/sundial
export STRIDER_DATA_DIR="$PWD/data"
export STRIDER_OUTPUT_DIR="$PWD/runs"
```

The [configuration guide](../configs/research/README.md) identifies the model
configuration, reference-bank builder and required data stores.
Keep the training, selection, calibration and test objects separate. Reserve
the test data for the final evaluation, and record the source files and object
assignments.

To prepare fresh stores for the included development configuration:

```bash
test ! -e "$STRIDER_DATA_DIR/ia_binary_full" || exit 1
test ! -e "$STRIDER_DATA_DIR/spectral_20k" || exit 1
test ! -e "$STRIDER_DATA_DIR/reference/roman_uncertainty_weighted.npz" || exit 1
strider prepare --config configs/research/ia_binary_full.yaml || exit 1
strider prepare --config configs/research/classes_test.yaml || exit 1
strider build-reference --config configs/research/uncertainty_reference.yaml || exit 1
```

Preparation replaces existing prepared files. When reusing verified stores,
skip preparation and verify their manifests, split counts and configuration.
The builder reads training data only. Use the preparation configurations shown
above: the bank configuration limits object counts and would not prepare the
full source store.

## Training, evaluation and calibration

1. Record the Git commit, environment, input manifests and resolved configuration.
2. Build and identify the reference bank before training. Record its checksum,
   full construction configuration and training-only provenance.
3. For a short development run, train with
   `strider train --config configs/research/reference_candidate_gate.yaml`.
   For an interrupted run, use the same configuration and `--resume`.
4. Evaluate the selected checkpoint on `selection` explicitly:

   ```bash
   strider evaluate --config configs/research/reference_candidate_gate.yaml --split selection
   ```

5. Compare models using the same selection objects and measurement views.
   Choose the model and checkpoint before fitting calibration.
6. Resolve or quantify the implementation issues in the
   [architecture guide](architecture.md#known-limitations), then freeze
   the accepted code, checkpoint, cohorts, calibration procedure and reporting rules.
7. Follow the [calibration commands](calibration.md#commands) on the reserved
   data, then perform the declared final evaluation. Record any test-data
   inspections made before the model and evaluation procedure were fixed.
8. Export a matching model package and preserve raw predictions and summaries.

`reference_candidate_gate.yaml` is a two-epoch development configuration.
It inherits a calibration default, so pass `--split selection` when evaluating
for model selection. Record the final training configuration before starting
a full run.

## Minimum run record

Keep these together for every reported result:

- Git commit and clean/dirty state;
- resolved configuration and SHA-256 digest;
- Python and dependency environment, hardware, precision and batch settings;
- source and prepared-data manifests;
- reference-bank format, metadata and checksum;
- checkpoint epoch and checksum;
- exact cohort identifiers, measurement views and random seeds; and
- raw predictions, calibration artifact and metric-generation command.

Before model release, publish a tested environment lock and a versioned manifest
covering the matching model, reference bank and calibration. Large scientific
assets belong in an archive or project store linked from the repository.
