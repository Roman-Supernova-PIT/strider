# STRIDER

**S**pectral **T**ime-se**R**ies **ID**entifi**E**r for **R**oman.

STRIDER is in development as a research tool for classifying transients and
estimating redshift from Roman-like prism observations. It works entirely from
measured observer-frame spectra, their reported uncertainties and their
observation dates. It does not require the transient's true redshift or
rest-frame phase.

The frozen calibrated STRIDER baseline remains the verified comparison. The
reference-based design in this repository is still undergoing matched
evaluation and is not presented as better or final.

## Quick check

STRIDER requires Python 3.11 or newer. The repository currently provides the
research pipeline and tests; a supported checkpoint download will follow only
after selection, calibration and final evaluation are complete.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The small temporal example checks the command-line path without simulation
files or a trained STRIDER checkpoint:

```bash
strider temporal-example \
  --epochs 2 \
  --training-objects 120 \
  --test-objects 40 \
  --output runs/temporal_example/summary.json
```

This is a smoke test, not a performance result.

## Measurements in, results out

One object contains one or more prism spectra. Each visit supplies:

- observer-frame wavelength in Angstrom;
- measured flux;
- reported flux uncertainty; and
- an observation date or equivalent observer-frame day coordinate.

The deployed boundary accepts no truth class, truth redshift, clean simulated
flux or simulated phase. Training labels may be used offline to build the fixed
simulation-derived reference bank and to calculate evaluation metrics.

STRIDER returns:

- a joint distribution over transient class and redshift;
- class probabilities and a redshift posterior with competing solutions kept;
- separate calibration and redshift-coverage information when fitted; and
- a separate measured-signal reliability result, including an explicit
  insufficient-spectral-information outcome.

## How it works

1. Sort the measured spectra into their observation sequence and resample each
   once onto a common observer-frame wavelength grid.
2. Form an inverse-variance accumulated spectrum from every available visit.
3. Keep two complementary views: the normalized full spectrum and its
   continuum-removed structure.
4. Compare both views with a fixed reference bank built from clean simulations
   in the training split only.
5. Scan every candidate class and redshift without supplying the object's true
   redshift or phase.
6. Select up to eight chronological temporal spectra and compare their measured
   evolution, relative brightness and timing with the candidate history.
7. Report the joint class-redshift result separately from calibration and
   measured-signal reliability.

The corrected wavelength-edge treatment leaves measured flux unchanged. A 5%
cosine taper changes only the influence of edge bins during matching. Apart from
the exact zero-weight endpoints, measured bins remain available unless their
relative precision is below the declared floating-point resolution floor.
Reference-bank format v3 records these semantics and rejects stale banks.

## Read next

- [`docs/start_here.md`](docs/start_here.md) follows one object through the tool.
- [`docs/architecture.md`](docs/architecture.md) defines the scientific and API
  boundaries.
- [`docs/data_and_models.md`](docs/data_and_models.md) records data, reference
  bank and model-package provenance.
- [`docs/research_workflow.md`](docs/research_workflow.md) gives the guarded run
  sequence and required records.
- [`docs/nersc.md`](docs/nersc.md) contains the concise NERSC path.
- [`docs/research_history.md`](docs/research_history.md) identifies earlier
  architectures and experiment records without presenting them as current.

## Repository map

| Path | Purpose |
|---|---|
| `src/strider/data/` | simulation input, native-bin preparation and measured-spectrum batching |
| `src/strider/atlas/roman_reference.py` | build and validate the training-only reference bank |
| `src/strider/model/coadd.py` | measurement-faithful inverse-variance accumulation and reliability weights |
| `src/strider/model/roman_reference.py` | full-spectrum, continuum-removed and observation-sequence matching |
| `src/strider/calibration/` | separate class, redshift-set and measured-signal calibration |
| `src/strider/deployment.py` | observer-frame inference from a checksummed model package |
| `src/strider/training/` | fitting, checkpoint selection and exact continuation |
| `src/strider/evaluation/` | predictions, controls and scientific diagnostics |
| `tests/` | scientific and implementation contracts |

## Roman context

STRIDER is being developed by members of the
[Roman Supernova Cosmology Project Infrastructure Team](https://www.romansnpit.com/).
Related community tools are available through the
[Roman Supernova PIT GitHub organization](https://github.com/Roman-Supernova-PIT).
The relevant mission context is NASA's
[Roman High-Latitude Time-Domain Survey](https://science.nasa.gov/mission/roman-space-telescope/high-latitude-time-domain-survey/).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Publication
details will be added when available.
