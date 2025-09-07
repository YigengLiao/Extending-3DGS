import torch
import tinycudann as tcnn

def ColorMLP(
    n_dir: int = 3,
    n_extra: int = 64,
    degree: int = 4,
    n_hidden_layers: int = 2,
    n_neurons: int = 64,
    n_out: int = 3
) -> tcnn.NetworkWithInputEncoding:

    encoding_cfg = {
        "otype": "Composite",
        "n_dims_to_encode": n_dir,
        "nested": [
            {"otype": "SphericalHarmonics", "degree": degree},
            {"otype": "Identity", "offset": n_dir, "n_dims_to_encode": n_extra, "scale": 1.0,}
        ]
    }

    network_cfg = {
        "otype": "FullyFusedMLP",
        "n_neurons": n_neurons,
        "n_hidden_layers": n_hidden_layers,
        "activation": "ReLU",
        "output_activation": "None"
    }

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims = n_dir + n_extra,
        n_output_dims = n_out,
        encoding_config = encoding_cfg,
        network_config = network_cfg
    )

    return model

def HashGridEncoder(
    n_input_dims: int = 3,
    n_levels: int = 16,
    n_features_per_level: int = 2,
    log2_hashmap_size: int = 19,
    base_resolution: int = 16,
    per_level_scale: float = 2.0
) -> tcnn.Encoding:

    encoding_config = {
        "otype": "Grid",
        "type": "Hash",
        "n_levels": n_levels,
        "n_features_per_level": n_features_per_level,
        "log2_hashmap_size": log2_hashmap_size,
        "base_resolution": base_resolution,
        "per_level_scale": per_level_scale
    }

    encoding = tcnn.Encoding(
        n_input_dims=n_input_dims,
        encoding_config=encoding_config
    )
    return encoding

def DensityMLP(
    input_dim: int = 34,
    n_hidden_layers: int = 2,
    n_neurons: int = 64,
    n_out_dims: int = 1,
    activation: str = "ReLU",
    output_activation: str = "None"
    ) -> tcnn.Network:
    return tcnn.Network(
        n_input_dims=input_dim,
        n_output_dims=n_out_dims,
        network_config={
            "otype": "FullyFusedMLP",
            "n_hidden_layers": n_hidden_layers,
            "n_neurons": n_neurons,
            "activation": activation,
            "output_activation": output_activation
        }
    )