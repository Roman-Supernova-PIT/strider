Installation
============

Install STRIDER from a clone. You need Python 3.11 or newer and PyTorch. The commands
below install the CPU build; for a GPU, install PyTorch as pytorch.org describes.

.. code-block:: console

    git clone https://github.com/Roman-Supernova-PIT/strider.git
    cd strider
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install .

For ECSV files, SMDC spectrum files, evidence maps and redshift curves, also install astropy, h5py and qp:
``pip install astropy h5py qp-prob``.

Then run ``strider check``. It downloads the model the first time, prints the versions, times the bundled example,
and checks its results against the stored ones:

.. code-block:: console

    strider check

The package is ``roman-snpit-strider``; you import ``strider`` and run ``strider``. Its version is read from git, so
install from a clone with its ``.git`` folder, or from the git URL. Without network access, as in some containers, set
``STRIDER_MODEL_DIR`` and run ``strider check`` while building the image, or load the model package file by path.
