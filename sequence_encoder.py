"""Attention-based User Sequence Encoding for IDDiff."""

from typing import Optional

import torch
import torch.nn as nn


class AttentionPFFNEncoder(nn.Module):
    """Encode spatio-temporally enhanced check-ins with attention and an FFN.

    This block corresponds to User Sequence Encoding in the Sequence-aware
    Trajectory Representation Module. It preserves the input shape ``(B, T, D)``
    so masked mean pooling can subsequently produce the whole-trajectory
    representation ``S_seq``.
    """

    def __init__(
        self, emb_dim: int, n_heads: int, ff_hidden: int, dropout: float
    ):
        super().__init__()
        # Multi-head self-attention models dependencies among historical check-ins.
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Residual layer normalization for the attention and FFN sublayers.
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        # Position-wise feed-forward network used by User Sequence Encoding.
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, emb_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return encoded check-in features with shape ``(B, T, D)``.

        ``key_padding_mask`` has shape ``(B, T)`` and marks ignored padding
        positions with ``True``.
        """
        # Attention sublayer with a residual connection.
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.dropout(attn_out))

        # Position-wise FFN sublayer with a residual connection.
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x


def mean_pool_with_mask(
    x: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Produce the whole-trajectory representation by masked mean pooling.

    Args:
        x: Encoded check-in features with shape ``(B, T, D)``.
        valid_mask: Binary mask with shape ``(B, T)``; one denotes a valid
            check-in and zero denotes padding.

    Returns:
        Whole-trajectory representations with shape ``(B, D)``.
    """
    mask_expanded = valid_mask.unsqueeze(-1)  # (B, T, 1)
    masked_x = x * mask_expanded
    sum_x = masked_x.sum(dim=1)  # (B, D)
    lengths = valid_mask.sum(dim=1).clamp(min=1)  # (B,)
    s_whole = sum_x / lengths.unsqueeze(-1)  # (B, D)
    return s_whole
