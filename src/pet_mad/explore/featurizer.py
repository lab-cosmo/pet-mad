import logging
import os
import warnings
import numpy as np
from typing import Literal, Optional
from urllib.request import urlretrieve

import metatensor.torch as mts
import metatomic.torch as mta
import torch
import vesin.metatomic as vesin_metatomic
from metatrain.utils.dtype import dtype_to_str
from platformdirs import user_cache_dir
from tqdm import tqdm

from .explorer_model import MADExplorer


warnings.filterwarnings(
    "ignore",
    message=("PET assumes that Cartesian tensors"),
)

VERSIONS = ["latest"]
PETMAD_MODEL = {
    "cache_filename": "pet-mad-latest.ckpt",
    "url": "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt",
}

PETMAD_EXPLORER = {
    "cache_filename": "pet-mad-explorer-{version}.ckpt",
    "url": "https://huggingface.co/sofiia-chorna/pet-mad-explorer/resolve/{branch}/pet-mad-explorer-latest.ckpt",
}

METADATA = mta.ModelMetadata(
    name="PET-MAD-Explorer",
    description="Exploration tool for PET-MAD model features upon SMAP projections",
    references={
        "model": ["http://arxiv.org/abs/2503.14118"],
    },
)


class PETMADFeaturizer:
    """
    Converts structures into low-dimensional projections using PET-MAD features,
    with dimensionality reduction based on sketch-map.

    Usage example:
        >>> import ase.io
        >>> import chemiscope
        >>> from pet_mad.explore import PETMADFeaturizer

        >>> # Load structures
        >>> frames = ase.io.read("dataset.xyz", ":")

        >>> # Create visualization
        >>> chemiscope.explore(
        ...     frames,
        ...     featurize=PETMADFeaturizer(version="latest")
        ... )
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
        batch_size: int = 1,
        progress_bar=tqdm,
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

        petmad_exp_path = self._get_model_path(
            model_type="explorer",
            version=version,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
        )

        petmad_path = self._get_model_path(
            model_type="model", checkpoint_path=pet_checkpoint_path, cache_dir=cache_dir
        )

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

        self.check_consistency = check_consistency
        self.device = device
        self.length_unit = length_unit
        self.batch_size = batch_size
        self.progress_bar = progress_bar

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

        outputs = []

        for i_sys in self.progress_bar(range(0, len(systems), self.batch_size)):
            batch_systems = systems[i_sys : i_sys + self.batch_size]
            batch_outputs = self.mad_explorer(
                batch_systems,
                options,
                check_consistency=self.check_consistency,
            )
            outputs.append(
                batch_outputs["features"].block().values.detach().cpu().numpy()
            )

        return np.concatenate(outputs)

    def _get_model_path(
        self,
        model_type: Literal["model", "explorer"],
        version: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cache_dir: str = "",
    ) -> str:
        if checkpoint_path is not None:
            logging.info(f"Loading model from checkpoint: {checkpoint_path}")
            return checkpoint_path

        if model_type == "explorer":
            cached_filename = PETMAD_EXPLORER["cache_filename"].format(version=version)
            branch = f"v{version}" if version != "latest" else "main"
            url = PETMAD_EXPLORER["url"].format(branch=branch)
        else:
            cached_filename = PETMAD_MODEL["cache_filename"]
            url = PETMAD_MODEL["url"]

        cached_path = os.path.join(cache_dir, cached_filename)

        if os.path.exists(cached_path):
            logging.info(f"Loading model from cache: {cached_path}")
            return cached_path

        logging.info("Downloading PET-MAD-Explorer model")
        path, _ = urlretrieve(url, cached_path)

        return path
