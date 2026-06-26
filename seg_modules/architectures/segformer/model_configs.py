from copy import deepcopy

_b0_config = {
    "channels": [32, 64, 160, 256],
    "decoder_dim": 256,
    "stages": [
        {"R": 8, "N": 1, "E": 8, "L": 2},
        {"R": 4, "N": 2, "E": 8, "L": 2},
        {"R": 2, "N": 5, "E": 4, "L": 2},
        {"R": 1, "N": 8, "E": 4, "L": 2},
    ],
}

_b1_config = {
    "channels": [64, 128, 320, 512],
    "decoder_dim": 256,
    "stages": [
        {"R": 8, "N": 1, "E": 8, "L": 2},
        {"R": 4, "N": 2, "E": 8, "L": 2},
        {"R": 2, "N": 5, "E": 4, "L": 2},
        {"R": 1, "N": 8, "E": 4, "L": 2},
    ],
}

_b2_config = {
    "channels": [64, 128, 320, 512],
    "decoder_dim": 768,
    "stages": [
        {"R": 8, "N": 1, "E": 8, "L": 3},
        {"R": 4, "N": 2, "E": 8, "L": 3},
        {"R": 2, "N": 5, "E": 4, "L": 6},
        {"R": 1, "N": 8, "E": 4, "L": 3},
    ],
}

_b3_config = {
    "channels": [64, 128, 320, 512],
    "decoder_dim": 768,
    "stages": [
        {"R": 8, "N": 1, "E": 8, "L": 3},
        {"R": 4, "N": 2, "E": 8, "L": 3},
        {"R": 2, "N": 5, "E": 4, "L": 18},
        {"R": 1, "N": 8, "E": 4, "L": 3},
    ],
}

_b4_config = {
    "channels": [64, 128, 320, 512],
    "decoder_dim": 768,
    "stages": [
        {"R": 8, "N": 1, "E": 8, "L": 3},
        {"R": 4, "N": 2, "E": 8, "L": 3},
        {"R": 2, "N": 5, "E": 4, "L": 27},
        {"R": 1, "N": 8, "E": 4, "L": 3},
    ],
}

_b5_config = {
    "channels": [64, 128, 320, 512],
    "decoder_dim": 768,
    "stages": [
        {"R": 8, "N": 1, "E": 4, "L": 3},
        {"R": 4, "N": 2, "E": 4, "L": 6},
        {"R": 2, "N": 5, "E": 4, "L": 40},
        {"R": 1, "N": 8, "E": 4, "L": 3},
    ],
}


_AVAILABLE_CONFIGS = {
    "b0": _b0_config,
    "b1": _b1_config,
    "b2": _b2_config,
    "b3": _b3_config,
    "b4": _b4_config,
    "b5": _b5_config,
}


def get_config(model_name: str):
    """Retrieves a deep copy of the model configuration."""
    if model_name.lower() not in _AVAILABLE_CONFIGS:
        raise ValueError(
            f"Model {model_name} is not available. Choose from {list(_AVAILABLE_CONFIGS.keys())}"
        )
    return deepcopy(_AVAILABLE_CONFIGS[model_name.lower()])
