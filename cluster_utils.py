import math
from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN


def compute_valid_lengths(padded_seq: torch.Tensor) -> torch.Tensor:
    """
    计算每个序列的有效长度（非零行计数）。
    padded_seq: (B, T, D)，零向量视为 Padding
    返回: (B,) 每个样本的有效长度
    """
    non_zero = (padded_seq.abs().sum(dim=-1) > 0).to(torch.long)  # (B, T)
    lengths = non_zero.sum(dim=1)  # (B,)
    return lengths


def sliding_windows_for_batch(x: torch.Tensor, window_size: int) -> List[List[torch.Tensor]]:
    """
    对 batch 中每个样本执行滑动窗口切分，仅切分有效长度范围。
    x: (B, T, D) 原始嵌入（包含 Padding 为全零向量）
    window_size: 滑动窗口大小
    返回: List[ List[Tensor] ]，外层按样本，内层为该样本的子序列张量列表，形状 (w, D)
    """
    B, T, D = x.shape
    lengths = compute_valid_lengths(x)  # (B,)
    batch_subseqs: List[List[torch.Tensor]] = []
    for i in range(B):
        li = int(lengths[i].item())
        if li <= 0:
            batch_subseqs.append([])
            continue
        effective = x[i, :li, :]  # (li, D)
        if li < window_size:
            batch_subseqs.append([effective])  # 长度不足，使用整段
        else:
            subs = []
            for s in range(li - window_size + 1):
                subs.append(effective[s : s + window_size, :])  # (window_size, D)
            batch_subseqs.append(subs)
    return batch_subseqs


def encode_subsequences(
    subseqs_per_sample: List[List[torch.Tensor]], encoder, device: str
) -> List[torch.Tensor]:
    """
    将子序列通过同样的注意力+PFFN+池化编码，得到每个样本的子序列向量集合。
    subseqs_per_sample: 每个样本的子序列列表
    encoder: AttentionPFFNEncoder
    device: 设备
    返回: List[Tensor]，每个样本对应形状 (N_i, D) 的子序列向量堆叠张量
    """
    encoded_list: List[torch.Tensor] = []
    for subs in subseqs_per_sample:
        if len(subs) == 0:
            encoded_list.append(torch.empty(0))
            continue
        # 对该样本的所有子序列逐个编码并池化
        vecs = []
        for s in subs:
            s = s.to(device).unsqueeze(0)  # (1, w, D)
            # 子序列内部无 Padding，因此 key_padding_mask 为全 False
            enc = encoder(s, key_padding_mask=None)  # (1, w, D)
            # 对序列维度做平均池化得到子序列嵌入
            v = enc.mean(dim=1)  # (1, D)
            vecs.append(v.squeeze(0))  # (D,)
        encoded_list.append(torch.stack(vecs, dim=0))  # (N_i, D)
    return encoded_list


def dbscan_cluster_centers(vectors: torch.Tensor, eps: float, min_samples: int) -> torch.Tensor:
    """
    对一个样本的子序列向量执行 DBSCAN，返回每个簇的中心（均值）。
    vectors: (N, D) 子序列嵌入
    返回: (K, D) 每个簇的均值；如果没有簇则返回 (1, D) 为所有向量的均值
    """
    if vectors.numel() == 0:
        return vectors  # 空张量，调用方需处理
    X = vectors.detach().cpu().numpy()  # (N, D)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_  # (N,)
    unique = set(labels.tolist())
    unique.discard(-1)  # 忽略噪声点
    centers = []
    if len(unique) == 0:
        # 无簇，回退为整体均值
        centers.append(X.mean(axis=0))
    else:
        for k in sorted(unique):
            cluster_points = X[labels == k]
            centers.append(cluster_points.mean(axis=0))
    centers_t = torch.tensor(np.stack(centers, axis=0), dtype=vectors.dtype, device=vectors.device)  # (K, D)
    return centers_t


def nearest_center_distance(s_whole: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算每个样本的 S_whole 与其对应簇中心的欧氏距离，选取最近中心。
    s_whole: (B, D)
    centers: List[Tensor]，每个元素 (K_i, D)
    返回: (nearest_center_stack, distances_stack)
    """
    device = s_whole.device
    nearest_centers = []
    dists = []
    for i in range(len(centers)):
        c = centers[i]  # (K_i, D)
        if c.numel() == 0:
            nearest_centers.append(torch.zeros_like(s_whole[i]))
            dists.append(torch.tensor(0.0, device=device))
            continue
        # 计算 (K_i,) 的欧氏距离
        diff = c - s_whole[i].unsqueeze(0)  # (K_i, D)
        dist = torch.norm(diff, dim=1)  # (K_i,)
        idx = torch.argmin(dist)
        nearest_centers.append(c[idx])
        dists.append(dist[idx])
    return torch.stack(nearest_centers, dim=0), torch.stack(dists, dim=0)

