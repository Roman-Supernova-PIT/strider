# STRIDER model packages

A current STRIDER release is a directory whose scientific components are
verified as one unit. Do not copy a calibration file between packages or point
the loader at a training run directory.

## Required files

| File | Purpose |
|---|---|
| `weights.pt` | Selected network checkpoint and its configuration digest |
| `config.resolved.yaml` | Complete architecture and preprocessing configuration |
| `model_info.json` | Model name, class set, redshift grid and release status |
| `calibration.json` | Class, redshift-set and source-sufficiency calibration |
| `redshift_grid.npy` | Candidate redshift coordinates |
| `wavelength_grid_angstrom.npy` | Observer-frame model wavelength grid |
| `preprocessing.yaml` | Human-readable measurement preprocessing contract |
| `onir_bank.npz` | Trained named-feature reference bank |
| `onir_features.yaml` | Named rest-frame feature definitions |
| `environment.json` | Training software and hardware provenance |
| `MODEL_CARD.md` | Model-specific use, evaluation and limitations |
| `SHA256SUMS` | Checksums covering every packaged artifact |

An evaluated release may also include `metrics.json`. The weights and
calibration remain unusable for calibrated claims unless all of the following
identifiers agree:

- resolved-configuration SHA-256;
- selected checkpoint epoch;
- model-package format;
- declared class names;
- wavelength grid; and
- redshift grid.

`strider check-model --model MODEL_DIRECTORY` performs these checks before the
model is used.

## Input contract

The deployment model receives only measured quantities:

- observer-frame wavelength in Angstrom;
- measured FLAM;
- measured FLAMERR;
- observer-frame dates or relative observer days; and
- optionally, a measured light-curve peak date when supported by the package.

Within each visit, FLAM is divided by the lower-quartile positive FLAMERR. This
is the same robust visit-level noise scale used during model training. Spectra
are then interpolated onto the packaged logarithmic observer-wavelength grid.
Bins outside the measured wavelength interval remain masked rather than being
treated as zero-valued measurements.

Simulation redshift, clean simulated flux and truth-derived rest phase are not
deployment inputs.

## Output contract

The JSON result records:

- raw and, when available, calibrated class probabilities;
- `z_STRIDER`, defined as the peak of the primary redshift basin;
- the primary and alternative redshift basins with their posterior masses;
- calibrated, possibly disconnected redshift sets;
- raw evidence sufficiency; and
- calibrated source probability and signal grade when fitted.

Raw conditional class confidence and calibrated source sufficiency answer
different questions. Users should inspect both before selecting an object for a
science sample.
