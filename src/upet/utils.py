import re
from pathlib import Path
from typing import Optional, Union
from urllib.parse import unquote

from huggingface_hub import hf_hub_download


hf_pattern = re.compile(
    r"(?P<endpoint>https://[^/]+)/"
    r"(?P<repo_id>[^/]+/[^/]+)/"
    r"resolve/"
    r"(?P<revision>[^/]+)/"
    r"(?P<filename>.+)"
)


def hf_hub_download_url(
    url: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Wrapper around `hf_hub_download` allowing passing the URL directly.

    Function is in inverse of `hf_hub_url`
    """

    match = hf_pattern.match(url)

    if not match:
        raise ValueError(f"URL '{url}' has an invalid format for the Hugging Face Hub.")

    endpoint = match.group("endpoint")
    repo_id = match.group("repo_id")
    revision = unquote(match.group("revision"))
    filename = unquote(match.group("filename"))

    # Extract subfolder if applicable
    parts = filename.split("/", 1)
    if len(parts) == 2:
        subfolder, filename = parts
    else:
        subfolder = None
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        cache_dir=cache_dir,
        revision=revision,
        token=hf_token,
        endpoint=endpoint,
    )


def __getattr__(name: str):
    # Deprecated: `align_dos` moved to `upet.ase.dos.utils`. Resolved lazily
    # (PEP 562) so importing `upet.utils` itself doesn't warn - only
    # accessing this specific, moved attribute does.
    if name == "align_dos":
        import warnings

        from .ase.dos.utils import align_dos

        warnings.warn(
            "upet.utils.align_dos is deprecated, use `from upet.ase.dos.utils "
            "import align_dos` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return align_dos
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
