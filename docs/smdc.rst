Running on the Roman SMDC
=========================

The SMDC runs Python inside the Roman SN PIT's Singularity containers. They include NumPy and SciPy but not PyTorch,
so STRIDER goes into an environment of your own, made once.

Set up once
-----------

Enter the current CPU container (its version changes, so list them first), then make the environment:

.. code-block:: console

    ls /data/snpit/*.sif
    singularity run /data/snpit/roman-snpit-env-cpu-<version>.sif /bin/bash
    python -m venv ~/strider-venv
    source ~/strider-venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "roman-snpit-strider[io] @ git+https://github.com/Roman-Supernova-PIT/strider.git"

The ``io`` extra adds h5py, which reading SMDC spectrum files needs.

Each session
------------

.. code-block:: console

    singularity run /data/snpit/roman-snpit-env-cpu-<version>.sif /bin/bash
    source ~/strider-venv/bin/activate
    strider check

``strider check`` downloads the model the first time, to ``~/.cache/strider`` or to ``$STRIDER_MODEL_DIR`` if you set
it, and confirms that it reproduces its stored example.

Classify SMDC spectrum files
----------------------------

Each file holds one visit. The files do not hold the date, so give the MJD of each:

.. code-block:: python

    import strider

    model = strider.load_model()
    transient = strider.io.read_smdc(["visit1.h5", "visit2.h5"], mjd=[61000.1, 61010.3])
    print(model.classify(transient))

For many transients, read each one and classify them together; the results are written as they go, and ``workers``
uses that many cores:

.. code-block:: python

    objects = {name: strider.io.read_smdc(files, mjd=dates) for name, (files, dates) in my_transients.items()}
    model.classify_many(objects, "results.parquet", workers=8)

The command line does not read SMDC files; ``strider.io.write_spectra(objects, "spectra.parquet")`` writes a table it
reads (:doc:`usage`).
