"""IDDIFF: Intent-aware Deterministic Diffusion for next-POI recommendation.

The implementation contains the Sequence-aware Trajectory Representation
Module, POI Distance Graph Encoder, and Intent Refinement Module. Existing
class and method names are retained to preserve training-script compatibility.
"""

import gol
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import DBSCAN

# Import the POI Distance Graph Encoder, deterministic refinement layer, and PFFN.
from layers import DisGraphRep, SDE_Diffusion, PFFN

"""Build a Boolean mask for valid check-ins in each padded trajectory."""
def Seq_MASK(lengths, max_len=None):
    lengths_shape = lengths.shape
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len).lt(lengths.unsqueeze(1))).reshape(lengths_shape)


class IDDIFF(nn.Module):
    """IDDIFF model with legacy class naming for checkpoint compatibility."""
    def __init__(self, n_user, n_poi, G_D: Data):
        super(IDDIFF, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.hid_dim = gol.conf['hidden']
        self.step_num = 1000
        self.skip_cluster_prob = 0.2
        self.alpha = gol.conf['alpha']

        # Spatio-Temporal Encoding: trainable POI, distance-interval, and time-interval embeddings.
        self.poi_emb = nn.Parameter(torch.empty(n_poi, self.hid_dim))
        self.delta_dis_embs = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        self.delta_time_embs = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        nn.init.xavier_normal_(self.poi_emb)
        nn.init.xavier_normal_(self.delta_dis_embs)
        nn.init.xavier_normal_(self.delta_time_embs)

        # The original sequence GCN is replaced by attention-based User Sequence Encoding.
        # self.seq_Rep = SeqGraphEncoder(self.hid_dim)

        # POI Distance Graph Encoder.
        self.dis_Rep = DisGraphRep(n_poi, self.hid_dim, G_D)
        # Intent Refinement Module with deterministic interpolation and denoising.
        self.SDEdiff = SDE_Diffusion(self.hid_dim, beta_min=gol.conf['beta_min'], beta_max=gol.conf['beta_max'], dt=gol.conf['dt'])

        self.CEloss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=gol.conf['dp'])

        # Sequence-aware Trajectory Representation Module: User Sequence Encoding.
        # Structure: Multi-head Attention -> LayerNorm -> PFFN -> Mean Pooling.
        self.seq_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn = nn.MultiheadAttention(embed_dim = self.hid_dim, num_heads=gol.conf['num_heads'], batch_first=True, dropout=0.2)
        self.seq_PFFN = PFFN(self.hid_dim, 0.2)

        # Intent Refinement Module: Location Archetype Generation attention block.
        self.geo_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn = nn.MultiheadAttention(embed_dim = self.hid_dim, num_heads=gol.conf['num_heads'], batch_first=True, dropout=0.2)
        self.geo_PFFN = PFFN(self.hid_dim, 0.2)

    def encode_sequence_block(self, embs_pad, seq_lengths):
        """
        Apply User Sequence Encoding and masked mean pooling to obtain S_seq.
        """
        # Ignore padded check-ins during attention and pooling.
        pad_mask = Seq_MASK(seq_lengths, max_len=embs_pad.size(1)) # [batch, max_len]

        # Multi-head self-attention captures dependencies among historical check-ins.
        Q = self.seq_layernorm(embs_pad)
        K = embs_pad
        V = embs_pad

        # Apply attention with masking; True entries are ignored padding positions.
        output, _ = self.seq_attn(Q, K, V, key_padding_mask=~pad_mask)

        output = output + Q
        output = self.seq_attn_layernorm(output)

        # Position-wise feed-forward refinement.
        output = self.seq_PFFN(output)

        # Masked mean pooling produces the whole-trajectory representation S_seq.
        mask = pad_mask.unsqueeze(-1).float() # [batch, max_len, 1]
        sum_embs = torch.sum(output * mask, dim=1) # [batch, hid_dim]
        cnt = torch.sum(mask, dim=1) # [batch, 1]
        cnt = torch.clamp(cnt, min=1e-9)

        S_pooled = sum_embs / cnt
        return S_pooled

    def get_sequence_mean(self, POI_embs, seqs):
        """Compute the pooled trajectory vector used as the diffusion terminus."""
        seq_lengths = torch.LongTensor([len(seq) for seq in seqs]).to(gol.device)
        seqs_pad = pad_sequence(seqs, batch_first=True, padding_value=0)
        embs_pad = POI_embs[seqs_pad]
        mask = Seq_MASK(seq_lengths, max_len=embs_pad.size(1)).unsqueeze(-1).float()
        sum_embs = torch.sum(embs_pad * mask, dim=1)
        cnt = torch.sum(mask, dim=1).clamp(min=1e-9)
        return sum_embs / cnt

    """Sequence-aware Trajectory Representation Module.

    It obtains the whole-trajectory representation, constructs local intent
    prototypes with sliding windows and DBSCAN, and selects the prototype most
    similar to the global representation as the user intent condition ``S_u``.
    """
    def SeqGraphRep(self, POI_embs, seqs):
        # User Sequence Encoding: pad full trajectories before attention.
        seq_lengths = torch.LongTensor([len(seq) for seq in seqs]).to(gol.device)
        seqs_pad = pad_sequence(seqs, batch_first=True, padding_value=0) # [batch, max_len]
        embs_pad = POI_embs[seqs_pad] # [batch, max_len, hid_dim]

        if gol.conf['dropout']:
            embs_pad = self.dropout(embs_pad)

        # Obtain the whole-trajectory representation S_seq.
        S_whole = self.encode_sequence_block(embs_pad, seq_lengths) # [batch, hid_dim]

        # Preserve the existing evaluation path, which uses the global representation.
        if not self.training:
            return S_whole

        # Preserve stochastic prototype-path skipping used by the original implementation.
        if torch.rand(1).item() < self.skip_cluster_prob:
            return S_whole

        # Intent Prototype Construction: sliding windows, shared encoding, and DBSCAN.
        window_size = gol.conf['window_size']
        guidance_w = gol.conf['guidance_w']
        dbscan_eps = gol.conf['dbscan_eps']
        dbscan_min_samples = gol.conf['dbscan_min_samples']

        S_final_list = []

        # Construct candidate intent prototypes independently for each user.
        for i in range(len(seqs)):
            user_seq = seqs[i] # Tensor of indices
            seq_len = len(user_seq)

            # Divide the trajectory into local subtrajectories.
            subsequences = []
            if seq_len < window_size:
                subsequences.append(user_seq)
            else:
                for j in range(seq_len - window_size + 1):
                    subsequences.append(user_seq[j : j + window_size])

            # Encode local subtrajectories with the shared User Sequence Encoder.
            sub_seqs_pad = pad_sequence(subsequences, batch_first=True, padding_value=0)
            sub_embs_pad = POI_embs[sub_seqs_pad]

            if gol.conf['dropout']:
                sub_embs_pad = self.dropout(sub_embs_pad)

            sub_lengths = torch.LongTensor([len(s) for s in subsequences]).to(gol.device)

            # Reuse the same attention-PFFN encoder as the full trajectory.
            sub_vecs = self.encode_sequence_block(sub_embs_pad, sub_lengths) # [num_subs, hid_dim]

            # Cluster local intent representations with DBSCAN.
            sub_vecs_np = sub_vecs.detach().cpu().numpy()

            if len(sub_vecs_np) > 0:
                clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(sub_vecs_np)
                labels = clustering.labels_

                # Use cluster centroids as candidate intent prototypes.
                unique_labels = set(labels)
                cluster_centers = []

                for label in unique_labels:
                    if label == -1: continue  # Exclude DBSCAN noise points.

                    indices = np.where(labels == label)[0]
                    center = np.mean(sub_vecs_np[indices], axis=0)
                    cluster_centers.append(center)

                if not cluster_centers:
                    # Fall back to the global representation if no valid prototype exists.
                    C_nearest = S_whole[i]
                else:
                    cluster_centers = torch.tensor(np.array(cluster_centers), device=gol.device, dtype=torch.float)

                    # Select the dominant intent prototype nearest to S_seq.
                    dists = torch.norm(cluster_centers - S_whole[i].unsqueeze(0), dim=1)
                    min_idx = torch.argmin(dists)
                    C_nearest = cluster_centers[min_idx]
            else:
                 # Defensive fallback for an unexpected empty local-intent set.
                 C_nearest = S_whole[i]

            # Form the prototype-guided user intent representation S_u.
            S_final_i = (1 - guidance_w) * S_whole[i] + guidance_w * C_nearest
            S_final_list.append(S_final_i)

        S_final = torch.stack(S_final_list, dim=0)
        return S_final

    """Location Archetype Generation in the Intent Refinement Module.

    ``S_u`` queries the visited geography-aware POI embeddings from ``R_V`` to
    produce the initial location prototype ``hat_L_u``.
    """
    def LocaGenerator(self, POI_embs, seqs, S_u):
        # Encode the global POI distance graph into geography-aware embeddings R_V.
        R_V = self.dis_Rep.encode(POI_embs)
        if gol.conf['dropout']:
            R_V = self.dropout(R_V)

        seq_lengths = torch.LongTensor([seq.size(0) for seq in seqs]).to(gol.device)
        R_V_seq = [R_V[seq] for seq in seqs]


        R_V_pad = pad_sequence(R_V_seq, batch_first=True, padding_value=0)
        pad_mask = Seq_MASK(seq_lengths)
        Q = self.geo_layernorm(S_u.detach().unsqueeze(1))
        K = R_V_pad
        V = R_V_pad

        output, att_weights = self.geo_attn(Q, K, V, key_padding_mask=~pad_mask)

        # The original implementation omits the query residual at this point.
        output = output.squeeze(1)
        output = self.geo_attn_layernorm(output)

        hat_L_u = self.geo_PFFN(output)  # Initial location prototype hat_L_u.
        return hat_L_u, R_V

    """Intent Structure-Preserving Diffusion.

    The deterministic forward path interpolates from ``hat_L_u`` to the pooled
    trajectory terminus, while reverse denoising uses ``S_u`` as the condition
    and returns the refined latent location preference ``L_u``.
    """
    def DiffGenerator(self, hat_L_u, S_u, seq_mean, target=None):
        local_embs = hat_L_u
        condition_embs = S_u.detach()

        # Reverse Denoising conditioned on the selected intent prototype S_u.
        L_u = self.SDEdiff.ReverseSDE_gener(local_embs, condition_embs, gol.conf['T'], x_end=seq_mean)

        loss_div = None
        if target is not None:  # Training phase.
            t_sampled = np.random.randint(1, self.step_num) / self.step_num

            # Forward Interpolation: x_t = (1 - t) * hat_L_u + t * seq_mean.
            perturbed_data = self.SDEdiff.ForwardSDE_diff(target, seq_mean, t_sampled)

            # Learn to reconstruct the initial location prototype from x_t.
            pred_x0 = self.SDEdiff.Est_score(perturbed_data, condition_embs, t_sampled)
            loss_div = torch.square(pred_x0 - target.detach()).mean()

        return L_u, loss_div



    """Compute recommendation loss and deterministic refinement loss."""
    def getTrainLoss(self, batch):
        usr, pos_lbl, exclude_mask, seqs, G_u, cur_time = batch

        R_v = self.poi_emb
        POI_embs = self.poi_emb
        if gol.conf['dropout']:
            POI_embs = self.dropout(POI_embs)

        seq_mean = self.get_sequence_mean(POI_embs, seqs)

        # Sequence-aware Trajectory Representation Module produces the intent condition S_u.
        S_u = self.SeqGraphRep(POI_embs, seqs)

        hat_L_u, R_V = self.LocaGenerator(POI_embs, seqs, S_u)
        L_u, loss_div = self.DiffGenerator(hat_L_u, S_u, seq_mean, target=hat_L_u)

        Y_predscore = self.alpha * torch.matmul(S_u, R_v.t()) + torch.matmul(L_u, R_V.t())

        loss_rec = self.CEloss(Y_predscore, pos_lbl)
        return loss_rec, loss_div


    def forward(self, seqs, G_u):
        """Rank all candidate POIs using sequential and refined spatial scores."""
        R_v = self.poi_emb
        POI_embs = self.poi_emb

        seq_mean = self.get_sequence_mean(POI_embs, seqs)
        S_u = self.SeqGraphRep(POI_embs, seqs)
        hat_L_u, R_V = self.LocaGenerator(POI_embs, seqs, S_u)

        # Average multiple refined preference samples during inference when requested.
        sample_num = gol.conf.get('sample_num', 1)
        if sample_num > 1:
            L_u_list = []
            for _ in range(sample_num):
                sample, _ = self.DiffGenerator(hat_L_u, S_u, seq_mean)
                L_u_list.append(sample)
            L_u = torch.stack(L_u_list).mean(dim=0)
        else:
            L_u, _ = self.DiffGenerator(hat_L_u, S_u, seq_mean)
        # End of inference-time preference averaging.

        '''
        Y_predscore = [batch_size, #POI]
        S_u         = [batch_size, hid_dim]
        R_v.T       = [hid_dim, #POI]
        L_u         = [batch_size, hid_dim]
        R_V.T       = [hid_dim, #POI]
        '''

        Y_predscore = self.alpha * torch.matmul(S_u, R_v.t()) + torch.matmul(L_u, R_V.t())
        return Y_predscore
