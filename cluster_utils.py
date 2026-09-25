"""Utilities for the IDDIFF Intent Prototype Construction stage.

The functions implement valid-length detection, sliding-window trajectory
partitioning, shared subsequence encoding, DBSCAN prototype construction, and
nearest-prototype selection with respect to the whole-trajectory representation.
"""

from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN


def compute_valid_lengths(padded_seq: torch.Tensor) -> torch.Tensor:
    """Return the number of non-padding check-ins in each padded sequence.

    Args:
        padded_seq: Tensor of shape ``(B, T, D)``. An all-zero row is treated
            as a padding position.

    Returns:
        Tensor of shape ``(B,)`` containing the valid trajectory lengths.
    """
    non_zero = (padded_seq.abs().sum(dim=-1) > 0).to(torch.long)  # (B, T)
    lengths = non_zero.sum(dim=1)  # (B,)
    return lengths


def sliding_windows_for_batch(
    x: torch.Tensor, window_size: int
) -> List[List[torch.Tensor]]:
    """Construct local subtrajectories for Intent Prototype Construction.

    Only valid, non-padding check-ins are partitioned. A trajectory shorter
    than ``window_size`` is retained as one local subtrajectory, following the
    fallback strategy described for IDDIFF.

    Args:
        x: Padded trajectory embeddings with shape ``(B, T, D)``.
        window_size: Number of check-ins in each local subtrajectory.

    Returns:
        A per-user list of subtrajectory tensors, each with shape ``(w, D)``.
    """
    _, _, _ = x.shape
    lengths = compute_valid_lengths(x)  # (B,)
    batch_subseqs: List[List[torch.Tensor]] = []
    for i in range(x.size(0)):
        valid_length = int(lengths[i].item())
        if valid_length <= 0:
            batch_subseqs.append([])
            continue

        effective = x[i, :valid_length, :]  # (valid_length, D)
        if valid_length < window_size:
            # Use the complete valid trajectory when no full window exists.
            batch_subseqs.append([effective])
        else:
            subtrajectories = []
            for start in range(valid_length - window_size + 1):
                subtrajectories.append(
                    effective[start : start + window_size, :]
                )  # (window_size, D)
            batch_subseqs.append(subtrajectories)
    return batch_subseqs


def encode_subsequences(
    subseqs_per_sample: List[List[torch.Tensor]], encoder, device: str
) -> List[torch.Tensor]:
    """Encode local subtrajectories with the shared User Sequence Encoder.

    Each local subtrajectory passes through the same attention-PFFN encoder as
    the full trajectory and is mean-pooled into a local intent representation.

    Args:
        subseqs_per_sample: Per-user local subtrajectory tensors.
        encoder: Shared ``AttentionPFFNEncoder`` instance.
        device: Target execution device.

    Returns:
        A list whose i-th tensor has shape ``(N_i, D)``.
    """
    encoded_list: List[torch.Tensor] = []
    for subtrajectories in subseqs_per_sample:
        if not subtrajectories:
            encoded_list.append(torch.empty(0))
            continue

        local_intent_vectors = []
        for subtrajectory in subtrajectories:
            subtrajectory = subtrajectory.to(device).unsqueeze(0)  # (1, w, D)
            # Local subtrajectories contain no internal padding positions.
            encoded = encoder(subtrajectory, key_padding_mask=None)  # (1, w, D)
            local_intent = encoded.mean(dim=1)  # (1, D)
            local_intent_vectors.append(local_intent.squeeze(0))  # (D,)

        encoded_list.append(torch.stack(local_intent_vectors, dim=0))  # (N_i, D)
    return encoded_list


def dbscan_cluster_centers(
    vectors: torch.Tensor, eps: float, min_samples: int
) -> torch.Tensor:
    """Construct candidate intent prototypes with DBSCAN.

    A prototype is the centroid of one DBSCAN cluster of local intent
    representations. If DBSCAN returns no valid cluster, the mean of all local
    representations is used as a stable fallback prototype.

    Args:
        vectors: Local intent representations with shape ``(N, D)``.
        eps: DBSCAN neighborhood radius.
        min_samples: Minimum number of samples required by DBSCAN.

    Returns:
        Candidate intent prototypes with shape ``(K, D)``.
    """
    if vectors.numel() == 0:
        return vectors  # The caller handles an empty trajectory explicitly.

    vector_array = vectors.detach().cpu().numpy()  # (N, D)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vector_array)
    labels = clustering.labels_  # (N,)
    cluster_ids = set(labels.tolist())
    cluster_ids.discard(-1)  # Exclude DBSCAN noise points.

    prototypes = []
    if not cluster_ids:
        # Fall back to one prototype when no density-based cluster is found.
        prototypes.append(vector_array.mean(axis=0))
    else:
        for cluster_id in sorted(cluster_ids):
            cluster_points = vector_array[labels == cluster_id]
            prototypes.append(cluster_points.mean(axis=0))

    return torch.tensor(
        np.stack(prototypes, axis=0),
        dtype=vectors.dtype,
        device=vectors.device,
    )  # (K, D)


def nearest_center_distance(
    s_whole: torch.Tensor, centers: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the dominant intent prototype nearest to each full trajectory.

    Args:
        s_whole: Whole-trajectory representations with shape ``(B, D)``.
        centers: Per-user candidate intent prototype tensors of shape
            ``(K_i, D)``.

    Returns:
        The selected dominant intent prototypes and their Euclidean distances.
    """
    device = s_whole.device
    nearest_centers = []
    distances = []
    for i, user_centers in enumerate(centers):
        if user_centers.numel() == 0:
            nearest_centers.append(torch.zeros_like(s_whole[i]))
            distances.append(torch.tensor(0.0, device=device))
            continue

        difference = user_centers - s_whole[i].unsqueeze(0)  # (K_i, D)
        user_distances = torch.norm(difference, dim=1)  # (K_i,)
        nearest_index = torch.argmin(user_distances)
        nearest_centers.append(user_centers[nearest_index])
        distances.append(user_distances[nearest_index])

    return torch.stack(nearest_centers, dim=0), torch.stack(distances, dim=0)
