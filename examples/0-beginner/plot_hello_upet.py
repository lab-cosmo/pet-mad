"""
Hello UPET
==========

A minimal introduction to the UPET package. Prints the installed version
and the public top-level API. More examples covering ASE molecular
dynamics, batched evaluation with ``metatrain``, LAMMPS input generation,
uncertainty quantification, and the PET-MAD-DOS electronic density of
states will be added here over time.
"""

import upet


print(f"UPET version: {upet.__version__}")

public_api = sorted(name for name in dir(upet) if not name.startswith("_"))
print("Public top-level API:")
for name in public_api:
    print(f"  - upet.{name}")
