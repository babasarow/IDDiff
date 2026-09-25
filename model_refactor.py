"""Auxiliary refactoring of the IDDIFF trajectory-intent pipeline.

This file isolates User Sequence Encoding and Intent Prototype Construction.
The production model remains in ``model.py``; this auxiliary class preserves
its existing interface and tensor names for compatibility.
"""

import torch
import torch.nn as nn

from config import ModelConfig
from sequence_encoder import AttentionPFFNEncoder, mean_pool_with_mask
from cluster_utils import (
    sliding_windows_for_batch,
    encode_subsequences,
    dbscan_cluster_centers,
    nearest_center_distance,
)


class IDDIFFRefactored(nn.Module):
    """Prototype-oriented implementation of trajectory intent extraction.

    The pipeline applies User Sequence Encoding to obtain a whole-trajectory
    representation, constructs local intent prototypes with sliding windows
    and DBSCAN, and returns a prototype-guided condition for a downstream
    Intent Refinement Module. The legacy class name is retained to avoid
    breaking external imports.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Sequence-aware Trajectory Representation Module: shared encoder.
        self.encoder = AttentionPFFNEncoder(
            emb_dim=cfg.emb_dim,
            n_heads=cfg.n_heads,
            ff_hidden=cfg.ff_hidden,
            dropout=cfg.dropout,
        )
        self.to(cfg.device)

    def forward(self, seq_embeddings: torch.Tensor) -> dict:
        """Extract full-trajectory and prototype-guided intent representations.

        Args:
            seq_embeddings: Padded POI embedding sequences with shape
                ``(B, T, D)``. All-zero rows are treated as padding.

        Returns:
            Intermediate representations from User Sequence Encoding and
            Intent Prototype Construction.
        """
        device = self.cfg.device
        x = seq_embeddings.to(device)  # (B, T, D)

        # Padding mask for multi-head self-attention; True means ignored.
        valid_mask = x.abs().sum(dim=-1) > 0  # (B, T)
        key_padding_mask = ~valid_mask  # (B, T)

        # User Sequence Encoding: attention, residual normalization, and FFN.
        encoded = self.encoder(x, key_padding_mask=key_padding_mask)  # (B, T, D)

        # Masked mean pooling produces the whole-trajectory representation.
        s_whole = mean_pool_with_mask(
            encoded, valid_mask.to(encoded.dtype)
        )  # (B, D)

        # Intent Prototype Construction: local subtrajectory generation.
        batch_subseqs = sliding_windows_for_batch(
            x, window_size=self.cfg.window_size
        )
        subseq_vectors = encode_subsequences(
            batch_subseqs, encoder=self.encoder, device=device
        )

        # Cluster local intent representations and use centroids as prototypes.
        centers_per_sample = []
        for vectors in subseq_vectors:
            centers = dbscan_cluster_centers(
                vectors=vectors.to(device),
                eps=self.cfg.dbscan_eps,
                min_samples=self.cfg.dbscan_min_samples,
            )  # (K, D) or fallback (1, D)
            centers_per_sample.append(centers)

        # Select the prototype nearest to the whole-trajectory representation.
        nearest_centers, _ = nearest_center_distance(
            s_whole, centers_per_sample
        )  # (B, D)

        # Retain the auxiliary implementation's original weighted combination.
        weight = self.cfg.guidance_w
        s_final = (1.0 - weight) * s_whole - weight * nearest_centers  # (B, D)

        return {
            "S_whole": s_whole,
            "S_final": s_final,
            "encoded_seq": encoded,
            "subseq_vectors": subseq_vectors,
            "centers_per_sample": centers_per_sample,
            "nearest_centers": nearest_centers,
        }

    def generate_with_diffusion(
        self, diff_generator, s_final: torch.Tensor, **kwargs
    ):
        """Pass the prototype-guided condition to an Intent Refinement Module.

        The downstream generator may expose either ``forward`` or ``generate``.
        ``s_final`` has shape ``(B, D)`` and serves as the intent condition.
        """
        if hasattr(diff_generator, "forward"):
            return diff_generator.forward(s_final, **kwargs)
        if hasattr(diff_generator, "generate"):
            return diff_generator.generate(s_final, **kwargs)
        raise AttributeError(
            "diff_generator must define either a forward or generate method"
        )
