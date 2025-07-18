from pet_mad.calculator import PETMADCalculator
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator, _ase_to_torch_data, _compute_ase_neighbors
from metatensor.torch.atomistic import ModelOutput, System, ModelEvaluationOptions, register_autograd_neighbors
from ase.build import bulk
from metatrain.utils.io import load_model
import torch

if __name__ == "__main__":
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    model = load_model("https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt")
    # model = model.export().module
    types, positions, cell, pbc = _ase_to_torch_data(atoms=atoms, dtype=torch.float32, device="cpu")
    system = System(
        types=types,
        positions=positions,
        cell=cell,
        pbc=pbc,
    )
    
    for options in model.requested_neighbor_lists():
        neighbors = _compute_ase_neighbors(
            atoms, options, dtype=torch.float32, device="cpu"
        )
        register_autograd_neighbors(
            system,
            neighbors,
            check_consistency=False
        )
        system.add_neighbor_list(options, neighbors)
    systems = [system]

    print(model.additive_models[0].weights['energy'].block().values.dtype)
    model(
        systems=systems,
        outputs={"energy": ModelOutput(per_atom=False)},
        selected_atoms=None,
    )
    print(model.additive_models[0].weights['energy'].block().values.dtype)

