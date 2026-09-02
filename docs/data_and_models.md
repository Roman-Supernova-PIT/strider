# Data and model provenance

STRIDER separates data by scientific role. These roles are part of the method,
not merely directory names.

| Role | Permitted use |
|---|---|
| Training | Fit model parameters and build the fixed simulation-derived reference bank |
| Selection | Compare candidate architectures and choose the frozen procedure |
| Calibration | Fit class, redshift-coverage and measured-signal calibration after selection |
| Test | One final evaluation after the model, calibration procedure and reporting rules are frozen |

Selection, calibration and test objects must never contribute spectra to the
reference bank. Runtime objects contain measured observer-frame spectra only;
clean simulated flux, true redshift and simulated phase remain outside the
deployed boundary.

## Prepared data

Raw simulations are converted to versioned HDF5 and Parquet stores by the
preparation commands. Those stores are too large and too source-specific for
Git. A reproducible run should record:

- the source simulation name, version and access location;
- the exact input file manifest and checksums;
- the split assignment and preparation configuration digest; and
- the prepared-store manifest and object counts.

The repository does not redistribute the current simulation products. They
must be obtained from their creators or an authorized project store.

## Reference bank

The active bank format is `strider-roman-spectral-reference-v3`. A bank records
its source split, configuration digest, class and phase grid, input manifest and
edge-weighting semantics. The v3 format is deliberately incompatible with
earlier banks that tapered measured flux rather than matching influence.

The bank is a fitted scientific artifact even though it has no gradient-trained
parameters. Publish it with a checksum, construction configuration and source
data citation. Do not put a full bank in Git.

## Model package

[`export_model_package`](../src/strider/model_package.py) writes a checksummed,
architecture-aware directory containing:

- model weights and the resolved configuration;
- the candidate redshift and observer-wavelength grids;
- preprocessing metadata and environment information;
- only the fixed assets required by that architecture;
- calibration and frozen-test metadata when they exist; and
- a model card and `SHA256SUMS` manifest.

A Roman-reference package contains `reference_bank.npz`; an ONIR package
contains its ONIR bank and catalog; a full-scan package carries neither. This
keeps deployment self-contained without preserving unrelated experimental
routes.

No supported STRIDER checkpoint or reference bank is released yet. Once the
current gate is complete, accepted artifacts should be deposited in a stable
archive with a version, DOI or persistent URL, checksums and the exact Git
commit. Large checkpoints, reference banks, prepared data and prediction tables
belong in that archive or project storage, not in this repository.
