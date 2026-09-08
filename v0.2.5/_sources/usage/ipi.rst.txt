.. _usage_ipi:

i-PI
====

.. note::

   Full UPET-with-i-PI documentation is work in progress.

UPET models can be driven by `i-PI <https://ipi-code.org/>`_ via the
``metatomic`` engine. See the `metatomic i-PI documentation
<https://docs.metatensor.org/metatomic/latest/engines/ipi.html>`_ for the
general workflow; exporting a UPET checkpoint with ``mtt export`` (see
:ref:`usage_metatrain`) produces a TorchScript model that can be plugged
directly into the i-PI driver.
