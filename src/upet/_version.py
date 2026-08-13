UPET_AVAILABLE_MODELS = [
    "pet-mad-xs",
    "pet-mad-s",
    "pet-omat-xs",
    "pet-omat-s",
    "pet-omat-m",
    "pet-omat-l",
    "pet-omat-xl",
    "pet-oam-l",
    "pet-oam-xl",
    "pet-omad-xs",
    "pet-omad-s",
    "pet-omad-l",
    "pet-omatpes-l",
    "pet-omol-s",
    "pet-omol-m",
    "pet-omol-l",
    "pet-mols-s",
    "pet-spice-s",
    "pet-spice-l",
]

UPET_NO_NC_SUPPORT_MODELS = [
    "pet-mad-s-v1.0.2",
    "pet-spice-s-v0.2.0",
    "pet-spice-l-v0.2.0",
    "pet-mols-s-v1.0.0",
    "pet-mols-s-v1.1.0",
]

# Models predicting non-conservative forces but no non-conservative stress, since
# they were trained on non-periodic data. Their stress is computed by
# backpropagation even when the non-conservative regime is requested.
UPET_NO_NC_STRESS_MODELS = [
    "pet-omol-s-v1.0.0",
    "pet-omol-m-v1.0.0",
    "pet-omol-l-v1.0.0",
]

UPET_UQ_SUPPORTED_MODELS = [
    "pet-mad-s-v1.0.2",
    "pet-mad-xs-v1.5.0",
    "pet-mad-s-v1.5.0",
    "pet-mols-s-v1.0.0",
    "pet-mols-s-v1.1.0",
]

DEPRECATED_MODELS: list[str] = []

# PET-MAD DOS
PET_MAD_DOS_LATEST_STABLE_VERSION = "1.0"
PET_MAD_DOS_AVAILABLE_VERSIONS = ["1.0"]
