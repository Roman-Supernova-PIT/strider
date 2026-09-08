# Using STRIDER

## Model availability

The model is being updated and tested. A supported trained model package is not
yet available to download. The Python interface below works with a model package
exported by STRIDER.

## Input format

For one transient, the example uses an `object.npz` file with four arrays:

| Array | Shape | Contents |
|---|---|---|
| `wavelength` | `(wavelength,)` | Shared observer-frame wavelength grid in Angstrom |
| `flux` | `(visits, wavelength)` | Measured spectral flux density, FLAM |
| `flux_error` | `(visits, wavelength)` | Reported one-sigma uncertainty in the same units as flux |
| `observer_time` | `(visits,)` | Observation dates in observer-frame days |

Use the same physical flux units across all visits and preserve their relative
brightness. Dates can be expressed as MJD or another consistent day coordinate.
The model uses time differences. True class, true redshift and simulated phase
are not inference inputs.

## Python inference

```python
import numpy as np
from strider.deployment import load_model_package

model = load_model_package("/path/to/model-package", device="cpu")
with np.load("object.npz", allow_pickle=False) as spectra:
    result = model.classify(
        wavelength=spectra["wavelength"],
        flux=spectra["flux"],
        flux_error=spectra["flux_error"],
        observer_time=spectra["observer_time"],
    )
print(result["classification"])
print(result["redshift"]["z_STRIDER"])
print(result["signal"])
```

The command-line interface currently supports training, evaluation and model
export. Analysing an input file uses the Python interface above; file-based
`classify` and `check-model` commands are not yet provided.

## Results

- `classification` contains class probabilities labelled `raw` or `calibrated`.
- `redshift` contains the redshift distribution and alternative solutions.
  `z_STRIDER` is the peak within the highest-density posterior basin. Calibrated
  redshift sets are included when available.
- `signal` reports signal reliability separately from class and redshift.
  Without a fitted signal calibration, source probability and grade are null.

Signal reliability is not a measure of redshift accuracy. The
[calibration guide](calibration.md) explains these quantities and their meanings.

## Model files and updates

A reference-based STRIDER model package contains the trained weights, resolved
configuration, reference bank, wavelength and redshift grids, preprocessing
metadata, model card and checksums. Calibration and evaluation records are
included when available.

Use the complete package and its stated software requirements. Keep the model
version with your results so that later model updates can be distinguished.
Weights, reference bank and calibration must belong to the same model release.

Released model packages will be linked here with their version, checksums and
matching source commit. Large model and data files are stored separately from
the Git repository.

## Training your own model

The [training guide](training.md) explains data preparation, model fitting,
evaluation and export. Simulation products must be obtained from their creators
or an authorized project store. The fixed reference bank is constructed from
training data only; selection, calibration and test objects must remain separate.
