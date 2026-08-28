# STRIDER

**S**pectral **T**ime-se**R**ies **ID**entifi**E**r for **R**oman.

STRIDER classifies transient spectra and estimates redshift from one or more
Roman-like prism observations. It keeps every visit distinct and returns a
joint distribution over transient class and redshift, together with a separate
assessment of whether the measured spectra contain enough source information
to use that distribution.

The repository currently includes the established 15-class STRIDER checkpoint
and several example objects. Support for the next self-contained model packages
is being prepared here; their final calibrated checkpoints and performance
tables will be published only after calibration and frozen evaluation.

## Install

STRIDER needs Python 3.11 or newer. A separate environment keeps it isolated
from your system Python:

```bash
git clone https://github.com/Roman-Supernova-PIT/strider.git
cd strider

conda create -n strider python=3.11
conda activate strider

python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the included model:

```bash
strider check-model
```

## Run the included model

```bash
strider classify examples/SN20088677_ou/spectrum_*.csv --top-k 3
```

```text
STRIDER  SN20088677_ou
──────────────────────────────────────────────────
Input      5 spectra, phase -15.0 to +15.0 d

Class      Ia       0.998
Redshift   z = 0.1273   68% [0.1173, 0.1685]   Two possible redshifts

Top 3      Ia     0.998
           IIL    0.001
           Iax    0.000
```

This bundled checkpoint uses STRIDER's established rest-phase interface. Its
text, NPZ and FITS inputs are documented by `strider classify --help` and in
the example folders.

Save a machine-readable result or diagnostic evidence map:

```bash
strider classify examples/SN20088677_ou/spectrum_*.csv \
  --output-json output/result.json \
  --plot output/evidence.png \
  --plot-evolution output/evidence.gif
```

## Run a current model package

Current STRIDER models are directories rather than a single weight file. A
package contains the weights, resolved architecture, wavelength and redshift
grids, named-feature bank, calibration, provenance and checksums. STRIDER
verifies these files before constructing the model.

Once a calibrated package is released, run it through the same command:

```bash
strider check-model --model /path/to/model-package

strider classify object.csv \
  --model /path/to/model-package \
  --output-json output/result.json \
  --plot output/evidence.png
```

Current packages do **not** use simulated redshift or truth-derived rest-frame
phase. Their simplest CSV input is:

```text
object_id,epoch,mjd,wavelength,flux,flux_error
roman-1,1,62000.0,7500.0,...,...
roman-1,1,62000.0,7600.0,...,...
roman-1,2,62012.0,7500.0,...,...
roman-1,2,62012.0,7600.0,...,...
```

- `wavelength` is observer-frame wavelength in Angstrom;
- `flux` is measured FLAM, in any consistent units;
- `flux_error` is its reported one-sigma uncertainty and is required;
- `mjd` or `observer_time` supplies one observer-frame date per visit; and
- `epoch` or `visit` distinguishes spectra in a combined table.

One file per visit is also accepted. Supply its dates on the command line if
they are not stored in the files:

```bash
strider classify epoch1.csv epoch2.csv \
  --time 62000.0 62012.0 \
  --model /path/to/model-package
```

The date coordinate may be absolute MJD or relative observer days. Only
differences from the first chronological visit enter the model. An optional
measured light-curve peak date can be supplied with `--peak-time` when the
released package declares that route. STRIDER never substitutes simulation
truth for a missing measurement.

Current model-package input supports CSV/text and NPZ, along with a static
evidence map. FITS ingestion and animated evidence accumulation remain under
migration; the established checkpoint retains its existing FITS and animation
support.

## Python

The same loader accepts both model formats:

```python
from strider import load_model

model = load_model("/path/to/model-package")
result = model.classify(
    wavelength=wavelength_angstrom,
    flux=flux_by_visit,
    flux_error=flux_error_by_visit,
    observer_time=mjd_by_visit,
)

result["classification"]["class"]
result["classification"]["p_Ia"]
result["redshift"]["z_STRIDER"]
result["signal"]["source_probability"]
```

For the included established checkpoint, use its historical interface:

```python
model = load_model("models/strider.pt")
result = model.classify(wavelength, flux, phase)
```

## Understanding the result

Current model packages distinguish three products:

1. **Classification** — calibrated class probabilities when the package
   contains matched calibration; otherwise explicitly labelled raw values.
2. **Redshift** — `z_STRIDER` is the peak of the primary posterior basin. The
   result retains alternative basins instead of hiding a distinct solution in
   one broad interval.
3. **Signal sufficiency** — a separate calibrated probability and grade for
   whether measurable source information is present. A confident conditional
   class result is not a substitute for this check.

Calibrated redshift regions may be disconnected when several solutions remain
plausible. The loader refuses a calibration artifact fitted to a different
configuration or checkpoint epoch.

## Models and status

| Model | Purpose | Public status |
|---|---|---|
| Included 15-class checkpoint | Established STRIDER classification and redshift interface | Available in `models/strider.pt` |
| Normal Ia / other | Ia selection and redshift recovery | Training and calibration in progress |
| Grouped transient classes | Broader transient grouping | Training and calibration in progress |
| Detailed transient taxonomy | Fifteen-class classification | Training and calibration in progress |

The included checkpoint covers `Ia`, `91bg`, `Iax`, `IIP`, `IIL`, `IIb`, `IIn`,
`Ib`, `Ic`, `Ic-BL`, `SLSN`, `TDE`, `ILOT`, `KN` and `PISN`.

STRIDER is trained for Roman-like prism spectra over approximately
7500–18000 Angstrom and redshifts between 0 and 3. Results from substantially
different instruments or wavelength coverage are outside the evaluated model
domain and should be treated with care.

## Development and citation

Run the public tests with:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check strider tests
```

If you use STRIDER, please cite Dixon et al. (2026). The complete reference and
released-model identifier will be added with the publication artifact.
