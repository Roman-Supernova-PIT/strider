# Research workflow and run records

The repository can currently repeat code-level checks and project runs when
the authorized simulation data are available. It does not yet ship a supported
checkpoint or a fully locked release environment.

## Local code check

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The `strider temporal-example` command in the main README is a small pipeline
smoke test. It is not a scientific benchmark.

## Guarded candidate sequence

1. Record the Git commit, environment, input manifest and resolved configuration.
2. Prepare the named training, selection, calibration and test roles without
   moving objects between them.
3. Build the reference bank from the training role only and record its checksum.
4. Fit the model and select its checkpoint using the configured selection rule.
5. Run the current two-epoch candidate on `selection` only:

   ```bash
   export STRIDER_CONFIG=configs/nersc/reference_candidate_gate.yaml
   bash nersc/evaluate_reference_candidate.sh
   ```

6. Compare it with the frozen calibrated baseline on the same cohort and
   measurement views. Do not describe it as better or final before that review.
7. If accepted, freeze the architecture, checkpoint, cohorts, calibration
   procedure, metrics and plotting rules.
8. Fit calibration once on `calibration`, then open `test` once for the final
   report. Preserve raw predictions as well as summaries.

The candidate configuration resolves exactly like the checkpoint's original
configuration. This allows evaluation without changing the recorded digest.
The evaluation script explicitly selects the reserved selection split.

## Minimum run record

Keep these together for every reported result:

- Git commit and clean/dirty state;
- resolved configuration and SHA-256 digest;
- Python and dependency environment;
- source and prepared-data manifests;
- reference-bank format, metadata and checksum;
- checkpoint epoch and checksum;
- exact cohort identifiers, measurement views and random seeds; and
- raw predictions, calibration artifact and metric-generation command.

Before the first public model release, add a platform-tested environment lock
and publish the data/reference/model artifact manifest. The current
`pyproject.toml` describes compatible dependencies, but it is not an exact
environment lock for exact reruns.
