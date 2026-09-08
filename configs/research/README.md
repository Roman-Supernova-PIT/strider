# Training configurations

These files set the model, training and data options. Set the data and output
locations in the [training guide](../../docs/training.md#data-preparation)
and use `strider <command> --config <recipe>` from the repository root.

| Recipe | Purpose |
|---|---|
| `reference_candidate_gate.yaml` | Two-epoch development configuration; evaluate with `--split selection` |
| `uncertainty_reference.yaml` | Build the uncertainty-weighted reference bank from training data |
| `ia_binary_full.yaml` | Prepare the full reference-source store |
| `classes_test.yaml` | Prepare the bounded 15-class development store |

The model is being updated and tested. Current limitations are recorded
in the [architecture guide](../../docs/architecture.md#known-limitations).
The remaining files support training configurations and automated tests.
To analyse observations, use the configuration supplied with the model package.
These training configurations do not include model weights or data.

Record the hardware, numerical precision, batch size, gradient accumulation
and worker settings. Changes to the training settings require a new run record
and evaluation.
