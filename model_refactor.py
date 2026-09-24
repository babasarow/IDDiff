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


class DiffDGMNRefactored(nn.Module):
    """
    重构后的模型片段：替换原 SeqGraphEncoder 与 BiSeqGCN，采用注意力+PFFN 编码；
    在得到 S_whole 后，执行滑动窗口、子序列编码、DBSCAN 聚类与负向引导，得到 S_final；
    最后将 S_final 作为条件向量，接入扩散模型 DiffGenerator（此处提供适配接口）。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # 序列编码器初始化
        self.encoder = AttentionPFFNEncoder(
            emb_dim=cfg.emb_dim,
            n_heads=cfg.n_heads,
            ff_hidden=cfg.ff_hidden,
            dropout=cfg.dropout,
        )
        self.to(cfg.device)

    def forward(self, seq_embeddings: torch.Tensor) -> dict:
        """
        seq_embeddings: (B, T, D) 输入的 POI 嵌入序列，Padding 为全零向量
        返回: 字典，包含 S_whole, S_final 以及中间调试信息
        """
        device = self.cfg.device
        x = seq_embeddings.to(device)  # (B, T, D)
        # 构造注意力的 key_padding_mask：True 表示忽略位置
        valid_mask = (x.abs().sum(dim=-1) > 0)  # (B, T)
        key_padding_mask = ~valid_mask  # (B, T)

        # 1) 注意力 + PFFN 编码
        encoded = self.encoder(x, key_padding_mask=key_padding_mask)  # (B, T, D)

        # 2) 有效位置上的平均池化得到 S_whole
        s_whole = mean_pool_with_mask(encoded, valid_mask.to(encoded.dtype))  # (B, D)

        # 3) 滑动窗口切分与子序列编码
        batch_subseqs = sliding_windows_for_batch(x, window_size=self.cfg.window_size)  # 每个样本的子序列列表
        subseq_vectors = encode_subsequences(batch_subseqs, encoder=self.encoder, device=device)  # 每个样本 (N_i, D)

        # 4) DBSCAN 聚类并计算簇中心
        centers_per_sample = []
        for vecs in subseq_vectors:
            centers = dbscan_cluster_centers(
                vectors=vecs.to(device),
                eps=self.cfg.dbscan_eps,
                min_samples=self.cfg.dbscan_min_samples,
            )  # (K, D) 或 (1, D)
            centers_per_sample.append(centers)

        # 5) 计算最近簇中心并进行负向引导
        nearest_centers, _ = nearest_center_distance(s_whole, centers_per_sample)  # (B, D)
        w = self.cfg.guidance_w
        s_final = (1.0 - w) * s_whole - w * nearest_centers  # (B, D)

        return {
            "S_whole": s_whole,
            "S_final": s_final,
            "encoded_seq": encoded,
            "subseq_vectors": subseq_vectors,
            "centers_per_sample": centers_per_sample,
            "nearest_centers": nearest_centers,
        }

    def generate_with_diffusion(self, diff_generator, s_final: torch.Tensor, **kwargs):
        """
        扩散模型接入适配：将 S_final 作为条件向量输入到 DiffGenerator。
        diff_generator: 扩散模型实例，需支持以下任一接口：
            - forward(cond, **kwargs)
            - generate(cond, **kwargs)
        s_final: (B, D) 条件向量
        kwargs: 额外参数（如步数、噪声级别等）
        返回: 扩散模型输出
        """
        if hasattr(diff_generator, "forward"):
            return diff_generator.forward(s_final, **kwargs)
        if hasattr(diff_generator, "generate"):
            return diff_generator.generate(s_final, **kwargs)
        raise AttributeError("diff_generator 不包含 forward 或 generate 方法")

