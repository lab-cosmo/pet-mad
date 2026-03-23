from unittest.mock import MagicMock, patch

from packaging.version import Version

from upet.calculator import UPETCalculator


def test_offline_checkpoint_loading_fallback():
    # Test fallback to 0.0.0 when version inference fails
    with (
        patch("upet.calculator.upet_get_size_to_load") as mock_get_size,
        patch("upet.calculator.upet_get_version_to_load") as mock_get_version,
        patch("upet._models.load_metatrain_model") as mock_load_model,
        patch("upet._models.get_upet_metadata") as mock_metadata,
        patch("upet.calculator.MetatomicCalculator") as mock_metatomic,
        patch("upet.calculator.user_cache_dir", return_value="/tmp"),
        patch("os.makedirs"),
    ):
        mock_loaded_raw_model = MagicMock()
        mock_exported_model = MagicMock()
        mock_load_model.return_value = mock_loaded_raw_model
        mock_loaded_raw_model.export.return_value = mock_exported_model

        mock_capabilities = MagicMock()
        mock_capabilities.outputs = {
            "energy": {},
            "forces": {},
            "stress": {},
            "virial": {},
        }
        mock_exported_model.capabilities.return_value = mock_capabilities

        # Checkpoint with non-matching name
        calc = UPETCalculator(model="pet-mad-s", checkpoint_path="custom_name.ckpt")

        mock_get_size.assert_not_called()
        mock_get_version.assert_not_called()
        mock_load_model.assert_called_with("custom_name.ckpt")

        # Verify version "0.0.0" was used in metadata call
        # args[2] is version in get_upet -> get_upet_metadata
        # Wait, get_upet calls get_upet_metadata(model, size, version)
        # We can inspect the save path to deduce the version used

        # The save call in UPETCalculator uses: pt_path = cache_dir + f"/{model}-{size}-v{version}.pt"
        # model="pet-mad", size="s"
        expected_save_path = "/tmp/pet-mad-s-v0.0.0.pt"
        mock_exported_model.save.assert_called_with(
            expected_save_path, collect_extensions=None
        )


def test_offline_checkpoint_loading_inference():
    # Test version inference from filename
    with (
        patch("upet.calculator.upet_get_size_to_load") as mock_get_size,
        patch("upet.calculator.upet_get_version_to_load") as mock_get_version,
        patch("upet._models.load_metatrain_model") as mock_load_model,
        patch("upet._models.get_upet_metadata") as mock_metadata,
        patch("upet.calculator.MetatomicCalculator") as mock_metatomic,
        patch("upet.calculator.user_cache_dir", return_value="/tmp"),
        patch("os.makedirs"),
    ):
        mock_loaded_raw_model = MagicMock()
        mock_exported_model = MagicMock()
        mock_load_model.return_value = mock_loaded_raw_model
        mock_loaded_raw_model.export.return_value = mock_exported_model

        mock_capabilities = MagicMock()
        mock_capabilities.outputs = {
            "energy": {},
            "forces": {},
            "stress": {},
            "virial": {},
        }
        mock_exported_model.capabilities.return_value = mock_capabilities

        # Checkpoint with matching name
        checkpoint_name = "pet-mad-s-v1.2.3.ckpt"
        calc = UPETCalculator(model="pet-mad-s", checkpoint_path=checkpoint_name)

        mock_load_model.assert_called_with(checkpoint_name)

        # Verify inferred version "1.2.3" was used
        expected_save_path = "/tmp/pet-mad-s-v1.2.3.pt"
        mock_exported_model.save.assert_called_with(
            expected_save_path, collect_extensions=None
        )


def test_online_loading_defaults():
    # Verify that without checkpoint, helpers ARE called
    with (
        patch(
            "upet.calculator.upet_get_size_to_load", return_value="s"
        ) as mock_get_size,
        patch(
            "upet.calculator.upet_get_version_to_load", return_value=Version("1.0.0")
        ) as mock_get_version,
        patch(
            "upet._models.hf_hub_download", return_value="downloaded_path"
        ) as mock_download,
        patch("upet._models.load_metatrain_model") as mock_load_model,
        patch("upet._models.get_upet_metadata") as mock_metadata,
        patch("upet.calculator.MetatomicCalculator") as mock_metatomic,
        patch("upet.calculator.user_cache_dir", return_value="/tmp"),
        patch("os.makedirs"),
    ):
        mock_loaded_raw_model = MagicMock()
        mock_exported_model = MagicMock()

        mock_load_model.return_value = mock_loaded_raw_model
        mock_loaded_raw_model.export.return_value = mock_exported_model

        mock_capabilities = MagicMock()
        mock_capabilities.outputs = {
            "energy": {},
            "forces": {},
            "stress": {},
            "virial": {},
        }
        mock_exported_model.capabilities.return_value = mock_capabilities

        calc = UPETCalculator(model="pet-mad-s")

        mock_get_size.assert_called()
        mock_get_version.assert_called()
