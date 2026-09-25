"""Configuration for the auxiliary IDDiff trajectory representation code."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hyperparameters for User Sequence Encoding and prototype construction."""

    # Sequence-aware Trajectory Representation Module.
    emb_dim: int = 128  # POI embedding dimension.
    n_heads: int = 4  # Number of multi-head self-attention heads.
    ff_hidden: int = 256  # Hidden dimension of the position-wise FFN.
    dropout: float = 0.1  # Dropout probability.

    # Intent Prototype Construction.
    window_size: int = 5  # Sliding-window length for local subtrajectories.
    guidance_w: float = 0.3  # Weight used by the auxiliary prototype-guidance path.
    dbscan_eps: float = 0.5  # DBSCAN neighborhood radius.
    dbscan_min_samples: int = 2  # Minimum DBSCAN cluster size.

    # Runtime environment.
    device: str = "cuda"  # Execution device: "cuda" or "cpu".


def get_default_config() -> ModelConfig:
    """Return the default configuration."""
    return ModelConfig()
