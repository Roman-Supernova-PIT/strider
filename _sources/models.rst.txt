Models
======

A model is a trained network, its brightness comparison's class light curves and its priors. It comes as one
file, a model package.

Released models
---------------

.. list-table::
   :header-rows: 1

   * - Model
     - Classes
     - Released
   * - ``strider-15class`` (the default)
     - 15
     - with Dixon et al. (in prep.)

A retrained model keeps its name and gets a new version.

``strider models`` lists the released models. ``strider.load_model()`` loads the default model, and
``strider.load_model("name")`` loads another. Each model is downloaded once, into ``$STRIDER_MODEL_DIR``, or into
``~/.cache/strider`` if that is not set.

``strider.load_model("test")`` loads the bundled test model. Its results are placeholders.

To load a package file, give its path: ``strider.load_model("path/to/package.tar.gz")``; to pin it, give its
checksum too: ``strider.load_model("path/to/package.tar.gz", sha256="...")``. Its unpacked directory works
too. If a model package is damaged, download it again.

The model card
--------------

Each model package carries its card, and ``strider models --card NAME`` prints it: the model's version and the STRIDER
versions it works with, its classes, redshift grid, how it works, its training simulations, the range it is valid
over, its priors, its running time, its known limits and how to cite it.

The first model, ``strider-15class``, was trained with no core-collapse supernovae below redshift about 0.65. Below
0.5 it has seen only SNe Ia, SNe Iax and kilonovae, so it cannot tell an SN Ia from a core-collapse supernova there,
and a result below 0.5 says so in its checks ("trained at this redshift: Ia, Iax, KN only"). Between 0.5 and 0.65 its
core-collapse classes rest on what it learned at higher redshift.
