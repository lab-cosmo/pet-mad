# High-throughput, batched evaluation of PET-MAD in ASE

While the traditional methods of the provided ASE calculator are appropriate to
run simulations, they are not optimal when the goal is to evaluate the model on
a large number of pre-determined structures. For that case, we provide a much
more efficient batched evaluator in the `compute_energy` method.

## Example

```python
import torch
from pet_mad.calculator import PETMADCalculator
from ase.build import bulk


device = "cuda" if torch.cuda.is_available() else "cpu"
calculator = PETMADCalculator(version="latest", device="cuda")

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
dataset = [atoms] * 100

# Split in batches of 10. Generally, the larger the batch size, the better.
batch_size = 10
batches = []
for i in range(0, len(dataset), batch_size):
    batch = dataset[i : i + batch_size]
    batches.append(batch)

all_energies = []
all_forces = []
for batch in batches:
    # Calculate the energies and forces for the batch
    results = calculator.compute_energy(batch, compute_forces_and_stresses=True)
    all_energies.extend(results["energy"])
    all_forces.extend(results["forces"])
```

## A final note

Compared to other universal atomistic models (and especially those trained on
the MPtrj dataset), PET-MAD can only evaluated structure contatining elements
with atomic numbers up to 86, except for Astatine (At). You might want to remove
those structures in advance, or catch possible exceptions during evaluation.
