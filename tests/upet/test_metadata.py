import pytest
from huggingface_hub import HfApi
from packaging.version import Version

from pet_mad._version import UPET_AVAILABLE_MODELS
from pet_mad.utils import get_upet_metadata


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_get_upet_metadata(model_name):
    hf_api = HfApi()
    repo_files = hf_api.list_repo_files("lab-cosmo/upet")
    files_in_models_folder = [f[7:] for f in repo_files if f.startswith("models/")]
    model, size = model_name.rsplit("-", 1)
    all_model_files = [
        f
        for f in files_in_models_folder
        if f.startswith(f"{model}-{size}-") and f.endswith(".ckpt")
    ]
    all_model_versions = [
        Version(f.split(f"{model}-{size}-")[1].split(".ckpt")[0])
        for f in all_model_files
    ]
    all_model_versions = sorted(set(all_model_versions))

    for version in all_model_versions:
        model, size = model_name.rsplit("-", 1)
        metadata = get_upet_metadata(model, size, version)

        if "mad" in model.lower():
            authors = [
                "Arslan Mazitov (arslan.mazitov@epfl.ch)",
                "Filippo Bigi",
                "Matthias Kellner",
                "Paolo Pegolo",
                "Davide Tisi",
                "Guillaume Fraux",
                "Sergey Pozdnyakov",
                "Philip Loche",
                "Michele Ceriotti (michele.ceriotti@epfl.ch)",
            ]
        else:
            authors = [
                "Filippo Bigi (filippo.bigi@epfl.ch)",
                "Arslan Mazitov (arslan.mazitov@epfl.ch)",
                "Paolo Pegolo",
                "Michele Ceriotti (michele.ceriotti@epfl.ch)",
            ]

        assert metadata.name == f"{model.upper()}-{size.upper()} v{version}"
        assert metadata.description == (
            r"A universal interatomic potential for advanced materials modeling "
            r"based on a Point-Edge Transformer (PET) architecture, and trained on "
            r"the {} dataset. Model size: {}".format(model.split("-")[1].upper(), size)
        )

        assert metadata.authors == authors
        assert metadata.references == {
            "architecture": ["https://arxiv.org/abs/2305.19302v3"],
            "model": ["http://arxiv.org/abs/2503.14118"],
        }
