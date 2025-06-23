import logging
import os
import warnings
from typing import Optional
from urllib.request import urlretrieve

import metatensor.torch as mts
import metatomic.torch as mta
import torch
import vesin.metatomic as vesin_metatomic
from metatrain.utils.dtype import dtype_to_str
from platformdirs import user_cache_dir

from .explorer_model import MADExplorer


warnings.filterwarnings(
    "ignore",
    message=("PET assumes that Cartesian tensors"),
)

VERSIONS = ["latest"]

PETMAD_BASE_URL = (
    "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"
)
PETMAD_EXP_BASE_URL = "https://huggingface.co/sofiia-chorna/pet-mad-explorer/resolve/{}/pet-mad-explorer-latest.ckpt"
PETMAD_CACHE_FILENAME = "pet-mad-latest.pt"
PETMAD_EXPLORER_CACHE_FILENAME = "pet-mad-explorer-{}.pt"
METADATA = mta.ModelMetadata(
    name="PET-MAD-Explorer",
    description="Exploration tool for PET-MAD model features upon SMAP projections",
    references={
        "model": ["http://arxiv.org/abs/2503.14118"],
    },
)


class PETMADFeaturizer:
    """
    The featurizer that converts structures into low-dimentional projections upon PET-MAD features.
    The projections are based on the sketch-map.

    The model is intended for exploratory analysis and visualization of the
    learned representations.

    Usage example:
        import ase.io

        import chemiscope
        from pet_mad.explore import PETMADFeaturizer

        # Read structures
        frames = ase.io.read("dataset.xyz", ":")

        # Generate visualisation of the dataset
        chemiscope.explore(
            frames,
            featurize=PETMADFeaturizer(version="latest")
        )
    """

    def __init__(
        self,
        version: str = "latest",
        checkpoint_path: Optional[str] = None,
        pet_checkpoint_path: Optional[str] = None,
        *,
        check_consistency=False,
        device=None,
        length_unit="Angstrom",
    ):
        """
        :param version: PET-MAD version to use. Supported version is "latest".
        :param checkpoint_path: path to a checkpoint file to load the exploration model from. If
            provided, the `version` parameter is ignored.
        :param pet_checkpoint_path: path to a petmad checkpoint file to use for the model from. If not
            provided, the latest checkpoint is fetched from HF
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If `None`, we will try
            the options in the model's `supported_device` in order.
        :param length_unit: unit of length used in the structures
        """

        if version not in VERSIONS:
            raise ValueError(
                f"Version {version} is not supported. Supported versions are {VERSIONS}"
            )

        cache_dir = user_cache_dir("pet-mad", "metatensor")
        os.makedirs(cache_dir, exist_ok=True)

        if checkpoint_path is not None:
            logging.info(
                f"Loading PET-MAD-Explorer model from checkpoint: {checkpoint_path}"
            )
            petmad_exp_path = checkpoint_path
        else:
            cached_filename = PETMAD_EXPLORER_CACHE_FILENAME.format(
                f"v{version}" if version != "latest" else "main"
            )
            petmad_exp_cached_path = os.path.join(cache_dir, cached_filename)

            if os.path.exists(petmad_exp_cached_path):
                logging.info(
                    f"Loading PET-MAD-Explorer model from cache: {petmad_exp_cached_path}"
                )
                petmad_exp_path = petmad_exp_cached_path
            else:
                logging.info("Downloading PET-MAD-Explorer model")
                path = PETMAD_EXP_BASE_URL.format(
                    f"v{version}" if version != "latest" else "main"
                )
                petmad_exp_path, _ = urlretrieve(path, petmad_exp_cached_path)

        if pet_checkpoint_path is not None:
            logging.info(
                f"Loading PET-MAD model from checkpoint: {pet_checkpoint_path}"
            )
            petmad_path = pet_checkpoint_path
        else:
            petmad_cached_path = os.path.join(cache_dir, PETMAD_CACHE_FILENAME)

            if os.path.exists(petmad_cached_path):
                logging.info(f"Loading PET-MAD model from cache: {petmad_cached_path}")
                petmad_path = petmad_cached_path
            else:
                logging.info("Downloading PET-MAD model")
                petmad_path, _ = urlretrieve(PETMAD_BASE_URL, petmad_cached_path)

        explorer = MADExplorer(petmad_path, device=device)
        explorer.load_checkpoint(petmad_exp_path)

        outputs = {"features": mta.ModelOutput(per_atom=True)}

        self.dtype = torch.float64

        capabilities = mta.ModelCapabilities(
            outputs=outputs,
            length_unit="angstrom",
            supported_devices=["cpu", "cuda"],
            dtype=dtype_to_str(self.dtype),
            interaction_range=0.0,
            atomic_types=explorer.get_atomic_types(),
        )

        self.mad_explorer = mta.AtomisticModel(explorer.eval(), METADATA, capabilities)
        self.mad_explorer.save("mad_explorer.pt")

        pt_path = os.path.join(cache_dir, f"pet-mad-explorer-{version}.pt")
        self.mad_explorer.save(pt_path)

        self.check_consistency = check_consistency
        self.device = device
        self.length_unit = length_unit

    def __call__(self, frames, environments):
        systems = mta.systems_to_torch(frames)
        vesin_metatomic.compute_requested_neighbors(
            systems,
            self.length_unit,
            self.mad_explorer,
        )

        systems = [s.to(dtype=self.dtype, device=self.device) for s in systems]

        if environments is not None:
            selected_atoms = mts.Labels(
                names=["system", "atom"],
                values=torch.tensor(
                    [(system, atom) for system, atom, _ in environments]
                ),
            )
        else:
            selected_atoms = None

        options = mta.ModelEvaluationOptions(
            length_unit=self.length_unit,
            outputs={"features": mta.ModelOutput(per_atom=True)},
            selected_atoms=selected_atoms,
        )

        outputs = self.mad_explorer(
            systems,
            options,
            check_consistency=self.check_consistency,
        )

        return outputs["features"].block().values.detach().cpu().numpy()
