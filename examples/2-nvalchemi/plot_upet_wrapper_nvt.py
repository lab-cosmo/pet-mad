"""
UPETWrapper: NVT molecular dynamics with nvalchemi-toolkit
=============================================================

:py:class:`~upet.nvalchemi.UPETWrapper` computes conservative forces and
stress from the energy via autograd and can be driven by any
`nvalchemi-toolkit <https://github.com/NVIDIA/nvalchemi-toolkit>`_
integrator. This example runs it through nvalchemi's Langevin (NVT)
integrator on a small water cluster. The neighbor list is wired
automatically from :py:attr:`~nvalchemi.models.base.ModelConfig.neighbor_config`
via :py:class:`~nvalchemi.hooks.NeighborListHook`.

.. note::

   This example requires the optional ``nvalchemi`` extra:
   ``pip install "upet[nvalchemi]"``. If it isn't installed, the example
   prints an install hint and exits without failing.
"""

import math

import torch


try:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import NVTLangevin
    from nvalchemi.dynamics.base import DynamicsStage
    from nvalchemi.dynamics.hooks._utils import KB_EV, kinetic_energy_per_graph
    from nvalchemi.hooks import NeighborListHook

    from upet.nvalchemi import UPETWrapper

    NVALCHEMI_AVAILABLE = True
except ImportError:
    NVALCHEMI_AVAILABLE = False

# %%
# Loading the model
# ------------------
# Fetch a small UPET model from HuggingFace. Loading a local checkpoint
# instead is a matter of passing ``checkpoint_path=...``.

if NVALCHEMI_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UPETWrapper.from_checkpoint(
        model="pet-mad-xs", version="1.5.0", device=device
    )

    # %%
    # Neighbor list: automatic wiring via ModelConfig
    # ------------------------------------------------
    # :py:attr:`~nvalchemi.models.base.ModelConfig.neighbor_config` encodes
    # the cutoff, list format (COO or MATRIX), and whether to use a
    # half-list. :py:class:`~nvalchemi.hooks.NeighborListHook` reads this
    # automatically and rebuilds the list before every model evaluation.

    neighbor_hook = NeighborListHook(
        model.model_config.neighbor_config,
        stage=DynamicsStage.BEFORE_COMPUTE,
    )

    # %%
    # Building a water cluster
    # --------------------------
    # 3 water molecules (9 atoms), with velocities drawn from a
    # Maxwell-Boltzmann distribution at 300 K.

    def _make_water_cluster(n_molecules: int = 3, seed: int = 0) -> AtomicData:
        torch.manual_seed(seed)
        positions_list = []
        atomic_numbers_list = []

        o_h_bond = 0.96  # Angstrom
        half_angle = math.radians(104.5 / 2)

        for i in range(n_molecules):
            ox = float(i) * 3.5
            o_pos = torch.tensor([ox, 0.0, 0.0])
            h1_pos = o_pos + o_h_bond * torch.tensor(
                [math.sin(half_angle), 0.0, math.cos(half_angle)]
            )
            h2_pos = o_pos + o_h_bond * torch.tensor(
                [-math.sin(half_angle), 0.0, math.cos(half_angle)]
            )
            positions_list.extend([o_pos, h1_pos, h2_pos])
            atomic_numbers_list.extend([8, 1, 1])  # O=8, H=1

        positions = torch.stack(positions_list) + 0.02 * torch.randn(
            len(positions_list), 3
        )
        n_atoms = len(atomic_numbers_list)
        masses = torch.tensor(
            [15.999 if z == 8 else 1.008 for z in atomic_numbers_list]
        )
        v_stds = (KB_EV * 300.0 / masses).sqrt().unsqueeze(-1)
        return AtomicData(
            positions=positions.float(),
            atomic_numbers=torch.tensor(atomic_numbers_list, dtype=torch.long),
            forces=torch.zeros(n_atoms, 3),
            energy=torch.zeros(1, 1),
            velocities=(v_stds * torch.randn(n_atoms, 3)).float(),
        )

    data = _make_water_cluster(n_molecules=3)
    batch = Batch.from_data_list([data], device=device)
    print(f"System: water cluster ({batch.num_nodes} atoms, 3 H2O) on {device}")

    # %%
    # NVT MD simulation
    # ------------------
    # 100 steps of NVTLangevin at 300 K. The neighbor hook fires at
    # BEFORE_COMPUTE to rebuild the neighbor list before each forward pass.

    nvt = NVTLangevin(
        model=model,
        dt=0.1,  # fs
        temperature=300.0,
        friction=0.5,
        n_steps=100,
        random_seed=99,
    )
    nvt.register_hook(neighbor_hook)

    print("\nRunning 100 NVT steps...")
    batch = nvt.run(batch)
    print(f"Run complete: {nvt.step_count} steps")

    # %%
    # Observables
    # ------------
    # Kinetic energy per graph (eV) -> temperature, via
    # ``2 KE = 3 N k_B T``.

    ke_per_graph = kinetic_energy_per_graph(
        velocities=batch.velocities,
        masses=batch.atomic_masses,
        batch_idx=batch.batch_idx,
        num_graphs=batch.num_graphs,
    )  # [B, 1]

    n_atoms_per_graph = batch.num_nodes_per_graph.float()  # [B]
    t_inst = (2.0 * ke_per_graph.squeeze(-1)) / (3.0 * n_atoms_per_graph * KB_EV)

    mean_energy = batch.energy.squeeze(-1).mean().item()
    mean_temperature = t_inst.mean().item()

    print("\nFinal observables:")
    print(f"  Mean energy : {mean_energy:+.4f} eV")
    print(f"  Inst. temp. : {mean_temperature:.1f} K  (target = 300 K)")
else:
    print(
        "This example requires the optional 'nvalchemi' extra: "
        "pip install 'upet[nvalchemi]'"
    )
