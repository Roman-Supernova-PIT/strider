# STRIDER

**S**pectral **T**ime-se**R**ies **ID**entifi**E**r for **R**oman.

STRIDER classifies supernova spectra and estimates their redshift. It works on a
single spectrum but is built for several epochs of the same transient: it keeps
the epochs distinct, learns how the spectrum evolves, and returns a joint
probability distribution over class and redshift.

The trained model and a few example objects are included here.

## Install

STRIDER needs Python 3.11 or newer. A separate environment keeps it isolated
from your system Python:

```bash
git clone https://github.com/mdixon741/strider.git
cd strider

conda create -n strider python=3.11
conda activate strider

python -m pip install --upgrade pip
python -m pip install -e .
```

If you already have an active Python 3.11 environment, skip the two `conda`
commands. The install includes CSV, text, NPZ and FITS input, along with the
plotting packages.

Check the model loaded:

```bash
strider check-model
```

## Run it

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

Add `--verbose` for the input coverage, the gold-Ia cut, and the redshift
detail. A few more example objects live under `examples/` — run them the same way.

## Input

The simplest input is a CSV or text table:

```text
wavelength flux flux_err phase
```

- `wavelength` — observed, in Angstrom
- `flux` — any units; STRIDER normalises internally
- `flux_err` — optional, used for weighting
- `phase` — rest-frame days from peak brightness (required)

One file per epoch, or a single table with a `phase` column. For a single
spectrum you can pass the phase on the command line instead:

```bash
strider classify spectrum.txt --phase -7
```

STRIDER also reads `.npz` and FITS files.

For time series with `flux_err`, plot how Ia versus non-Ia confidence changes
with cumulative spectral signal-to-noise:

```bash
strider classify series.npz --plot-confidence confidence.png
```

## What you get back

- the type, with a ranked list and a calibrated probability for each class
- `z_STRIDER` — the posterior-median redshift — with 68/90/95% intervals
- how much to trust that redshift: reliable, uncertain, or two possible values
- which epochs and wavelengths were used

Two notes on the output:

- **"two possible redshifts"** means a second candidate the tight interval hides —
  prefer the reliable ones when you need a redshift you can trust.
- **gold-Ia** is a strict purity cut for cosmology samples; a confident Ia need
  not be gold.

Save the result as JSON, or draw an evidence map:

```bash
strider classify examples/SN20088677_ou/spectrum_*.csv \
  --output-text output/output.txt \
  --output-json output/result.json \
  --plot output/evidence.png \
  --plot-evolution output/evidence.gif \
  --plot-epochs output/timeseries
```

`output.txt` is the concise overall result shown in the terminal. The PNG shows
the final evidence using every supplied spectrum. The GIF adds the spectra in
phase order, while the `timeseries` folder contains the same cumulative evidence
map as a separate PNG after every epoch.

Run `strider classify --help` for the phase, redshift-prior and wavelength
controls. To run on the Roman SMDC, see [`deploy/smdc`](deploy/smdc/README.md).

## Python

```python
from strider import load_model

model = load_model("models/strider.pt")
out = model.classify(wavelength, flux, phase)   # phase in rest-frame days

out["strider_class"]   # 'Ia'
out["p_Ia"]
out["z_STRIDER"]
```

## Where it works

STRIDER was trained on Roman-like prism spectra over 7500–18000 Å and
0 < z < 3, and it reads redshift from where a supernova's features land in that
window. It is most reliable when the strong features sit inside that range.
Spectra that fall largely outside it, or come from a very different instrument,
are outside what the model has seen — treat those results with care.

## Model

The included `strider-15class` checkpoint covers:

`Ia`, `91bg`, `Iax`, `IIP`, `IIL`, `IIb`, `IIn`, `Ib`, `Ic`, `Ic-BL`, `SLSN`,
`TDE`, `ILOT`, `KN`, `PISN`.

## Citation

If you use STRIDER, please cite Dixon et al. (2026). *(Reference to follow on
publication.)*
