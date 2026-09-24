import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionPFFNEncoder(nn.Module):
    """
    序列编码器：移除 GAT/GCN，采用 MultiheadAttention -> LayerNorm -> PFFN 的结构。
    输入形状: (B, T, D)，其中 T 为序列长度，D 为嵌入维度；Padding 为全零向量。
    输出形状: (B, T, D)，保持与输入相同的批次和序列维度，用于后续池化。
    """

    def __init__(self, emb_dim: int, n_heads: int, ff_hidden: int, dropout: float):
        super().__init__()
        # 多头注意力，使用 batch_first 便于 (B, T, D)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        # 层归一化
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        # 前馈网络 (Position-wise FFN)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, emb_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D) 输入序列
        key_padding_mask: (B, T) 为 True 的位置将被注意力忽略（对应 Padding）
        返回: (B, T, D) 编码后的序列特征
        """
        # 注意力子层
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.dropout(attn_out))
        # 前馈子层
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x


def mean_pool_with_mask(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    对序列维度进行平均池化，仅对有效位置（非 Padding）进行平均。
    x: (B, T, D)
    valid_mask: (B, T) 取值为 {0,1}，1 表示有效位置
    返回: (B, D) S_whole
    """
    # 扩展 mask 以匹配 x 的最后一维
    mask_expanded = valid_mask.unsqueeze(-1)  # (B, T, 1)
    masked_x = x * mask_expanded
    sum_x = masked_x.sum(dim=1)  # (B, D)
    lengths = valid_mask.sum(dim=1).clamp(min=1)  # (B,)
    s_whole = sum_x / lengths.unsqueeze(-1)  # (B, D)
    return s_whole

