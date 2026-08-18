"""Expected model capabilities, asserted against what the library reports."""

# Models whose checkpoints do not expose `non_conservative_*` outputs.
UPET_NO_NC_SUPPORT_MODELS = frozenset(
    {
        "pet-mad-s-v1.0.2",
        "pet-spice-s-v0.2.0",
        "pet-spice-l-v0.2.0",
        "pet-mols-s-v1.0.0",
        "pet-mols-s-v1.1.0",
    }
)


def supports_non_conservative(model_name, version) -> bool:
    """Whether ``model_name`` at ``version`` is expected to provide NC outputs."""
    return f"{model_name}-v{version}" not in UPET_NO_NC_SUPPORT_MODELS


def non_conservative_error_message(
    model_name, version, non_conservative, variant="energy"
) -> str:
    """The error `UPETCalculator` is expected to raise for an unsupported model."""
    return (
        f"`non-conservative={non_conservative}` option is not available "
        f"for the model {model_name} v{version}, and a target variant "
        f"`{variant}`. Please choose another `non-conservative` option, "
        "use another target variant, switch to a conservative regime "
        "or choose another model."
    )
