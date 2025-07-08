from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch as mts
import metatomic.torch as mta
import torch

from metatrain.utils.io import load_model as load_metatrain_model
from metatrain.pet import PET

from ._modules import MLPProjector, TorchStandardScaler


class MADExplorer(torch.nn.Module):
    """
    Metatomic model projecting the last-layer features from PET-MAD into a
    low-dimensional space.

    The model is intended for exploratory analysis and visualization of the learned
    representations.

    :param model: path to a saved PET-MAD checkpoint or an in-memory instance
    :param input_dim: dimensionality of the input PET-MAD features for projector
    :param output_dim: target low dimensionality for the projected embeddings
    :param device: cpu or cuda
    :param features_output: key to access the PET-MAD feature output.
        mtt::aux::energy_last_layer_features is used by default
    """

    def __init__(
        self,
        model: Union[str, Path, PET],
        input_dim: int = 1024,
        output_dim: int = 3,
        device: Optional[Union[str, torch.device]] = "cpu",
        features_output: str = "mtt::aux::energy_last_layer_features",
    ):
        super().__init__()

        self.device = device
        self.features_output = features_output

        if isinstance(model, (str, Path)):
            self.pet = load_metatrain_model(model)
        else:
            self.pet = model

        self.pet = self.pet.to(self.device)
        self.dtype = next(self.pet.parameters()).dtype

        self.projector = MLPProjector(input_dim, output_dim).to(self.device)
        self.feature_scaler = TorchStandardScaler().to(device)
        self.projection_scaler = TorchStandardScaler().to(device)

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

        systems = [s.to(self.dtype, self.device) for s in systems]

        per_atom = outputs["features"].per_atom
        pet_requested_outputs = {
            self.features_output: mta.ModelOutput(per_atom=per_atom)
        }

        if selected_atoms is not None:
            selected_atoms = selected_atoms.to(self.device)

        features = self._get_features(systems, pet_requested_outputs, selected_atoms)

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

    def _get_features(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> torch.Tensor:
        """
        Compute embeddings for the given systems using the PET-MAD model.

        For per-atom features, it concatenates mean and standard deviation of
        features across atoms
        """

        output = self.pet(systems, outputs, selected_atoms)
        features = output[self.features_output]

        if selected_atoms is not None:
            features = mts.slice(features, "samples", selected_atoms)

        if outputs[self.features_output].per_atom:
            mean = mts.mean_over_samples(features, "atom")
            mean_vals = torch.cat([block.values for block in mean.blocks()], dim=0)

            std = mts.std_over_samples(features, "atom")
            std_vals = torch.cat([block.values for block in std.blocks()], dim=0)

            descriptors = torch.cat([mean_vals, std_vals], dim=1)
        else:
            descriptors = features.block().values

        if descriptors.shape[1] != self.projector.input_dim:
            raise ValueError(
                f"Expected input dim for projector: {self.projector.input_dim}, "
                "got: {descriptors.shape[1]}"
            )

        return descriptors.detach()

    def load_checkpoint(self, path: Union[str, Path]):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.projector.load_state_dict(checkpoint["projector_state_dict"])

        self.feature_scaler.mean = checkpoint["feature_scaler_mean"]
        self.feature_scaler.std = checkpoint["feature_scaler_std"]
        self.projection_scaler.mean = checkpoint["projection_scaler_mean"]
        self.projection_scaler.std = checkpoint["projection_scaler_std"]
