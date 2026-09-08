# SPDX-License-Identifier: BSD-3-Clause
"""Integration with `nvalchemi-toolkit
<https://github.com/NVIDIA/nvalchemi-toolkit>`_.

This subpackage is only usable when the optional ``nvalchemi`` extra is
installed::

    pip install "upet[nvalchemi]"
"""

try:
    from .wrapper import UPETWrapper
except ImportError as e:
    raise ImportError(
        "upet.nvalchemi requires the optional 'nvalchemi-toolkit' dependency. "
        "Install it with: pip install 'upet[nvalchemi]'"
    ) from e

__all__ = ["UPETWrapper"]
