# Migrating from PET-MAD to UPET

This guide provides instructions for users who are transitioning from the
**PET-MAD** package to the new **UPET** package. The UPET package is a
successor of PET-MAD, offering enhanced functionalities and a broader scope of
models for atomistic simulations trained on popular datasets for atomistic machine
learning. The latest version of the PET-MAD package is `1.4.3`, and users are
encouraged to migrate to UPET for continued support and new features.

Migrating to UPET involves:

- Updating the package name from `pet_mad` to `upet`.
- Change in the calculator name from `PETMADCalculator` to `UPETCalculator`.
- Adjusting the model name while initializing the calculator to reflect the new
  naming conventions in UPET.

## Step 1: Update Package Import
Replace all instances of:

```python
from pet_mad.calculator import PETMADCalculator
```

with:

```python
from upet.calculator import UPETCalculator
```

## Step 2: Update Calculator Initialization
When initializing the calculator, update the model name to match the new
naming conventions in UPET. For example, if you were using the PET-MAD v1.0.2,
for example, you would change:

```python
calculator = PETMADCalculator(version='1.0.2')
```

to:

```python
calculator = UPETCalculator(model='pet-mad-s', version='1.0.2')
```

## Step 3: Update the Hugging Face Model Repository URL
If you are loading models directly from the Hugging Face model repository,
update the URL from:

```bash
https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt # For version 1.0.2
https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt # For version 1.1.0
```

to:

```bash
https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.0.2.ckpt # For version 1.0.2
https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.1.0.ckpt # For version 1.1.0
```

In the case of any issues or further assistance needed during the migration
process, please refer to the UPET documentation or open an issue on the UPET
GitHub repository.