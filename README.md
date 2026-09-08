# STRIDER

**S**pectral **T**ime-se**R**ies **ID**entifi**E**r for **R**oman.

STRIDER is a research tool for classifying transients and estimating redshift
from Roman-like prism spectra. It uses measured fluxes, their uncertainties and
observation dates.

The model is being updated and tested. The code and an example are available
now; a trained model package will be provided after evaluation is complete.

## Installation

STRIDER requires Python 3.11 or newer. From the repository directory, run:

```bash
python -m pip install .
```

## Example

This small example uses artificial time series to check the installation.
It needs no simulation files or trained model:

```bash
strider example \
  --epochs 2 \
  --training-objects 120 \
  --test-objects 40 \
  --output runs/example/summary.json
```

## Inputs and outputs

For each transient, provide one or more spectra with:

- observer-frame wavelength in Angstrom;
- measured flux and its uncertainty; and
- the observation date in days.

STRIDER estimates class probabilities and a redshift distribution, including
alternative redshift solutions. It also reports a separate signal-reliability
score. Calibrated probabilities and uncertainty intervals are included when
calibration is available in the model package.

The [usage guide](docs/data_and_models.md) describes the input format and Python
interface. Analysing observations requires an exported trained model package.

## How it works

STRIDER combines all available spectra into an accumulated spectrum and compares
it with a fixed reference bank across candidate classes and redshifts. Up to
eight observations also provide information about spectral changes, relative
brightness and timing. The model does not receive the transient's true redshift
or rest-frame phase.

## Documentation

- [Using STRIDER](docs/data_and_models.md): input arrays, Python example and model files.
- [Model](docs/architecture.md): calculations and current limitations.
- [Training and evaluation](docs/training.md): preparing data, fitting a model and checking results.
- [Calibration](docs/calibration.md): probabilities, redshift intervals and signal reliability.

## About and citation

STRIDER is being developed by members of the
[Roman Supernova Cosmology Project Infrastructure Team](https://www.romansnpit.com/).
Citation details are in [CITATION.cff](CITATION.cff); publication information will
be added when available.
