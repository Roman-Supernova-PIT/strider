# Running on the Roman SMDC

The SMDC runs Python through Singularity containers. They include NumPy and
SciPy but not PyTorch, so STRIDER needs a small personal virtualenv with CPU
PyTorch — set up once.

## Get STRIDER onto SMDC

While the repository is private, copy the bundle across from your machine and
unpack it:

```bash
scp strider_smdc_<date>.tar.gz roman:~/
```

Then, on SMDC:

```bash
rm -rf ~/strider
mkdir ~/strider
tar xzf ~/strider_smdc_<date>.tar.gz -C ~/strider
```

Once the repository is public you can `git clone` it instead.

## One-time setup

Find the current container (the version changes), enter it, and make the venv:

```bash
ls /data/snpit/*.sif
singularity run /data/snpit/roman-snpit-env-cpu-<version>.sif /bin/bash

python -m venv ~/strider-venv
source ~/strider-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu

cd ~/strider
pip install -e ".[plot]"
```

The `[plot]` extra adds matplotlib for evidence maps; leave it off if you do not
need plots.

## Each session

```bash
singularity run /data/snpit/roman-snpit-env-cpu-<version>.sif /bin/bash
source ~/strider-venv/bin/activate
cd ~/strider

strider check-model
strider classify examples/SN20088677_ou/spectrum_*.csv --output-json output/result.json --plot output/evidence.png
```

`check-model` should print `strider-15class`. From here it is the same `strider`
command as anywhere else — see the main [README](../../README.md) for input
formats, redshift priors, and the full JSON output.
