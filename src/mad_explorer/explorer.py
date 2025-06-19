from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch as mts
import metatomic.torch as mta
import torch
from torch import nn

from src.mad_explorer.mlp_projector import MLPProjector
from src.mad_explorer.scaler import TorchStandardScaler


class MADExplorer(nn.Module):
    """
    Metatomic wrapper model for extracting last-layer features from a PET-MAD and
    projecting them into a low-dimensional space using an MLP.

    The model is intended for exploratory analysis and visualization of the
    learned representations.

    :param mtt_model: path to a saved PET-MAD checkpoint or an in-memory instance
    :param mlp_checkpoint: path to a saved MLP checkpoint including projector weights and scalers
    :param extensions_directory: path to model extensions (if any)
    :param check_consistency: whether to verify consistency between model and system inputs
    :param input_dim: dimensionality of the input PET-MAD features for projector
    :param output_dim: target low dimensionality for the projected embeddings
    :param device: cpu or cuda
    :param features_output: which features of PET-MAD to use 
    """

    def __init__(
        self,
        mtt_model: Union[str, Path, mta.AtomisticModel],
        mlp_checkpoint: str,
        check_consistency: bool = False,
        input_dim: int = 1024,
        output_dim: int = 3,
        extensions_directory: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = "cpu",
        features_output: str = "mtt::aux::energy_last_layer_features",
    ):
        super().__init__()

        self.check_consistency = check_consistency
        self.device = device

        if isinstance(mtt_model, (str, Path)):
            self.petmad = mta.load_atomistic_model(
                mtt_model, extensions_directory=extensions_directory
            )
        else:
            self.petmad = mtt_model

        capabilities = self.petmad.capabilities()

        if features_output not in capabilities.outputs:
            raise ValueError(f"this model does not have a '{features_output}' output")
        else:
            self.features_output = features_output

        if capabilities.dtype == "float32":
            self.dtype = torch.float32
        else:
            assert capabilities.dtype == "float64"
            self.dtype = torch.float64

        self.projector = MLPProjector(input_dim, output_dim).to(self.device)
        self.feature_scaler = TorchStandardScaler().to(device)
        self.projection_scaler = TorchStandardScaler().to(device)

        if mlp_checkpoint:
            self.load_checkpoint(mlp_checkpoint)

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> Dict[str, mts.TensorMap]:
        if list(outputs.keys()) != ["features"]:
            raise ValueError(
                f"`outputs` keys ({', '.join(outputs.keys())}) contain unsupported "
                "keys. Only 'features' is supported"
            )

        per_atom = outputs["features"].per_atom
        if per_atom and selected_atoms is None:
            raise ValueError("Selected atoms must be provided for per-atom features")

        length_unit = self.petmad.capabilities().length_unit
        options = mta.ModelEvaluationOptions(
            length_unit=length_unit,
            outputs={self.features_output: mta.ModelOutput(per_atom=per_atom)},
            selected_atoms=selected_atoms,
        )

        systems = [s.to(self.dtype, self.device) for s in systems]
        features = self._get_descriptors(systems, options, per_atom)

        if self.feature_scaler.mean is not None and self.feature_scaler.std is not None:
            features = self.feature_scaler.transform(features)

        with torch.no_grad():
            projections = self.projector(features)

        if (
            self.projection_scaler.mean is not None
            and self.projection_scaler.std is not None
        ):
            projections = self.projection_scaler.inverse_transform(projections)

        num_atoms = projections.size(0)
        num_projections = projections.size(1)

        sample_labels = mts.Labels(
            "system", torch.arange(num_atoms, device=self.device).reshape(-1, 1)
        )
        prop_labels = mts.Labels(
            "projection",
            torch.arange(num_projections, device=self.device).reshape(-1, 1),
        )

        if projections.dtype != self.dtype:
            projections = projections.type(self.dtype)

        block = mts.TensorBlock(
            values=projections,
            samples=sample_labels,
            components=[],
            properties=prop_labels,
        )

        tensor_map = mts.TensorMap(
            keys=mts.Labels("_", torch.tensor([[0]], device=self.device)),
            blocks=[block],
        )

        return {"features": tensor_map}

    def _get_descriptors(
        self,
        systems: List[mta.System],
        options: mta.ModelEvaluationOptions,
        per_atom: bool,
    ) -> torch.Tensor:
        """
        Compute embeddings for the given systems using the PET-MAD model.

        For per-atom features, it concatenates mean and standard deviation of
        features across atoms
        """

        output = self.petmad(
            systems,
            options,
            check_consistency=self.check_consistency,
        )
        features = output[self.features_output]

        if per_atom:
            mean = mts.mean_over_samples(features, "atom")
            mean_vals = torch.cat([block.values for block in mean.blocks()], dim=0)

            std = mts.std_over_samples(features, "atom")
            std_vals = torch.cat([block.values for block in std.blocks()], dim=0)

            descriptors = torch.cat([mean_vals, std_vals], dim=1)
        else:
            descriptors = features.block().values

        if descriptors.shape[1] != self.projector.input_dim:
            raise ValueError(
                f"Expected input dim for projector: {self.projector.input_dim}, got: {descriptors.shape[1]}"
            )

        return descriptors.detach()

    def get_atomic_types(self) -> List[int]:
        return self.petmad.capabilities().atomic_types

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, weights_only=False)
        self.projector.load_state_dict(checkpoint["projector_state_dict"])

        self.feature_scaler.mean = checkpoint["feature_mean"].to(self.device)
        self.feature_scaler.std = checkpoint["feature_std"].to(self.device)

        self.projection_scaler.mean = checkpoint["projection_mean"].to(self.device)
        self.projection_scaler.std = checkpoint["projection_std"].to(self.device)

    def save_checkpoint(self, path: str):
        checkpoint = {
            "projector_state_dict": self.projector.state_dict(),
            "feature_mean": self.feature_scaler.mean,
            "feature_std": self.feature_scaler.std,
            "projection_mean": self.projection_scaler.mean,
            "projection_std": self.projection_scaler.std,
        }
        torch.save(checkpoint, path)


model = MADExplorer("pet-mad-latest.pt", mlp_checkpoint="mtt_projection_model.pt")

metadata = mta.ModelMetadata(
    name="mad-explorer",
    description="Exploration tool for PET-MAD model features upon SMAP projections",
    authors=["TODO"],
    references={
        "architecture": ["https://arxiv.org/abs/2305.19302v3"],
        "model": ["http://arxiv.org/abs/2503.14118"],
        "implementation": ["https://doi.org/10.1073/pnas.1108486108"],
    },
)

outputs = {
    "features": mta.ModelOutput(per_atom=True),
}

capabilities = mta.ModelCapabilities(
    outputs=outputs,
    length_unit="angstrom",
    supported_devices=["cpu", "cuda"],
    dtype="float64",
    interaction_range=0.0,
    atomic_types=model.get_atomic_types(),
)

mad_explorer = mta.AtomisticModel(model.eval(), metadata, capabilities)
mad_explorer.save("mad_explorer.pt")
