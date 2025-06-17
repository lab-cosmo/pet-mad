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
    Metatomic wrapper model for that extracts PET-MAD last-layer features and
    projects them into a low-dimensional space using an MLP.

    The model is intended for exploratory analysis and visualization of the
    learned representations.

    :param mtt_model: path to a saved PET-MAD model or an instance of a loaded
        model
    :param extensions_directory: path to model extensions
    :param check_consistency: whether to verify consistency between model and
        system inputs
    :param input_dim : dimensionality of the input PET-MAD features
    :param output_dim: target low dimensionality for the projected features
    :param device: device on which to run the model
    """

    def __init__(
        self,
        mtt_model: Union[str, Path, mta.AtomisticModel],
        mlp_checkpoint: Optional[str] = None,
        extensions_directory: Optional[str] = None,
        check_consistency: bool = False,
        input_dim: int = 1024,
        output_dim: int = 3,
        device: Optional[Union[str, torch.device]] = "cpu",
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

        self.output_name = "mtt::aux::energy_last_layer_features"
        if self.output_name not in capabilities.outputs:
            raise ValueError(f"this model does not have a '{self.output_name}' output")

        if capabilities.dtype == "float32":
            self.dtype = torch.float32
        else:
            assert capabilities.dtype == "float64"
            self.dtype = torch.float64

        self.projector = MLPProjector(input_dim, output_dim).to(self.device)
        self.feature_scaler = TorchStandardScaler().to(device)
        self.projection_scaler = TorchStandardScaler().to(device)

        if mlp_checkpoint:
            checkpoint = torch.load(mlp_checkpoint, weights_only=False)
            self.projector.load_state_dict(checkpoint["projector_state_dict"])

            self.feature_scaler.mean = checkpoint["feature_mean"].to(device)
            self.feature_scaler.std = checkpoint["feature_std"].to(device)

            self.projection_scaler.mean = checkpoint["projection_mean"].to(device)
            self.projection_scaler.std = checkpoint["projection_std"].to(device)

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

        if not outputs["features"].per_atom:
            raise NotImplementedError(
                "Model uses per-atom features to get mean and std features"
            )

        if selected_atoms is None:
            raise ValueError("MADExplorer requires 'selected_atoms' to be provided")

        length_unit = self.petmad.capabilities().length_unit
        options = mta.ModelEvaluationOptions(
            length_unit=length_unit,
            outputs={
                self.output_name: mta.ModelOutput(per_atom=selected_atoms is not None)
            },
            selected_atoms=selected_atoms,
        )

        systems = [s.to(self.dtype, self.device) for s in systems]
        features = self._get_descriptors(systems, options)

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
        self, systems: List[mta.System], options: mta.ModelEvaluationOptions
    ) -> torch.Tensor:
        """
        Compute embeddings for the given systems using the PET-MAD model.

        The method computes per-atom mean and std of features and returns as a
        combined tensor.
        """

        output = self.petmad(
            systems,
            options,
            check_consistency=self.check_consistency,
        )
        features = output[self.output_name]

        if options.selected_atoms is not None:
            mean = mts.mean_over_samples(features, "atom")
            mean_vals = torch.cat([block.values for block in mean.blocks()], dim=0)

            std = mts.std_over_samples(features, "atom")
            std_vals = torch.cat([block.values for block in std.blocks()], dim=0)

            combined_features = torch.cat([mean_vals, std_vals], dim=1)
            return combined_features.detach()
        else:
            return features.block().values.detach()

    def get_atomic_types(self) -> List[int]:
        return self.petmad.capabilities().atomic_types

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
