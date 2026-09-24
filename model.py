import gol
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import DBSCAN

# Import necessary layers. SeqGraphEncoder is removed as per requirement.
from layers import DisGraphRep, SDE_Diffusion, PFFN

'''Identify the actual length and padding of the user check-in trajectory sequence'''
def Seq_MASK(lengths, max_len=None): 
    lengths_shape = lengths.shape  
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len).lt(lengths.unsqueeze(1))).reshape(lengths_shape)


class DiffDGMN(nn.Module):
    def __init__(self, n_user, n_poi, G_D: Data):
        super(DiffDGMN, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.hid_dim = gol.conf['hidden']
        self.step_num = 1000
        self.skip_cluster_prob = 0.2
        self.alpha = gol.conf['alpha']

        # Initialize all parameters (POI embeddings, distance/time embeddings)
        self.poi_emb = nn.Parameter(torch.empty(n_poi, self.hid_dim))
        self.delta_dis_embs = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        self.delta_time_embs = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        nn.init.xavier_normal_(self.poi_emb)
        nn.init.xavier_normal_(self.delta_dis_embs)
        nn.init.xavier_normal_(self.delta_time_embs)

        # Removed SeqGraphEncoder (GAT/GCN) as requested
        # self.seq_Rep = SeqGraphEncoder(self.hid_dim)
        
        self.dis_Rep = DisGraphRep(n_poi, self.hid_dim, G_D)
        self.SDEdiff = SDE_Diffusion(self.hid_dim, beta_min=gol.conf['beta_min'], beta_max=gol.conf['beta_max'], dt=gol.conf['dt'])

        self.CEloss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=gol.conf['dp']) 

        # Initialize Module (A) Sequence Encoder
        # Structure: MultiheadAttention -> LayerNorm -> PFFN
        self.seq_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn = nn.MultiheadAttention(embed_dim = self.hid_dim, num_heads=gol.conf['num_heads'], batch_first=True, dropout=0.2)
        self.seq_PFFN = PFFN(self.hid_dim, 0.2)

        # Initialize Module (B) DisGraphRep and Module (C) LocaGenerator
        self.geo_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn = nn.MultiheadAttention(embed_dim = self.hid_dim, num_heads=gol.conf['num_heads'], batch_first=True, dropout=0.2)
        self.geo_PFFN = PFFN(self.hid_dim, 0.2)

    def encode_sequence_block(self, embs_pad, seq_lengths):
        """
        Helper method to apply Attention -> LayerNorm -> PFFN -> MeanPooling
        """
        # Create padding mask
        pad_mask = Seq_MASK(seq_lengths, max_len=embs_pad.size(1)) # [batch, max_len]
        
        # Self-Attention
        Q = self.seq_layernorm(embs_pad)
        K = embs_pad
        V = embs_pad
        
        # Apply attention with masking
        # key_padding_mask: True for values to be ignored
        output, _ = self.seq_attn(Q, K, V, key_padding_mask=~pad_mask)
        
        output = output + Q
        output = self.seq_attn_layernorm(output)
        
        # PFFN
        output = self.seq_PFFN(output)
        
        # Mean Pooling (excluding padding)
        mask = pad_mask.unsqueeze(-1).float() # [batch, max_len, 1]
        sum_embs = torch.sum(output * mask, dim=1) # [batch, hid_dim]
        cnt = torch.sum(mask, dim=1) # [batch, 1]
        cnt = torch.clamp(cnt, min=1e-9)
        
        S_pooled = sum_embs / cnt
        return S_pooled

    def get_sequence_mean(self, POI_embs, seqs):
        seq_lengths = torch.LongTensor([len(seq) for seq in seqs]).to(gol.device)
        seqs_pad = pad_sequence(seqs, batch_first=True, padding_value=0)
        embs_pad = POI_embs[seqs_pad]
        mask = Seq_MASK(seq_lengths, max_len=embs_pad.size(1)).unsqueeze(-1).float()
        sum_embs = torch.sum(embs_pad * mask, dim=1)
        cnt = torch.sum(mask, dim=1).clamp(min=1e-9)
        return sum_embs / cnt

    '''
    SeqGraphRep: Refactored Sequence Encoder
    Input: POI embeddings sequence (indices list)
    Return: Sequence Encoding S_final (conditioned with negative guidance)
    '''
    def SeqGraphRep(self, POI_embs, seqs):
        # 1. Encode whole sequences
        # Convert list of seqs to padded tensor
        seq_lengths = torch.LongTensor([len(seq) for seq in seqs]).to(gol.device)
        seqs_pad = pad_sequence(seqs, batch_first=True, padding_value=0) # [batch, max_len]
        embs_pad = POI_embs[seqs_pad] # [batch, max_len, hid_dim]

        if gol.conf['dropout']:
            embs_pad = self.dropout(embs_pad)

        # Apply Attention + PFFN + Pooling to get S_whole
        S_whole = self.encode_sequence_block(embs_pad, seq_lengths) # [batch, hid_dim]

        # In evaluation / prediction, always use S_whole as condition
        if not self.training:
            return S_whole

        # In training, with probability skip_cluster_prob, skip clustering
        if torch.rand(1).item() < self.skip_cluster_prob:
            return S_whole

        # 2. Sliding Window & Subsequence Clustering
        window_size = gol.conf['window_size']
        guidance_w = gol.conf['guidance_w']
        dbscan_eps = gol.conf['dbscan_eps']
        dbscan_min_samples = gol.conf['dbscan_min_samples']
        
        S_final_list = []
        
        # Process each user individually for clustering
        for i in range(len(seqs)):
            user_seq = seqs[i] # Tensor of indices
            seq_len = len(user_seq)
            
            # Sliding window slicing
            subsequences = []
            if seq_len < window_size:
                subsequences.append(user_seq)
            else:
                for j in range(seq_len - window_size + 1):
                    subsequences.append(user_seq[j : j + window_size])
            
            # Encode subsequences
            # Prepare batch of subsequences for this user
            sub_seqs_pad = pad_sequence(subsequences, batch_first=True, padding_value=0)
            sub_embs_pad = POI_embs[sub_seqs_pad]
            
            if gol.conf['dropout']:
                sub_embs_pad = self.dropout(sub_embs_pad)
                
            sub_lengths = torch.LongTensor([len(s) for s in subsequences]).to(gol.device)
            
            # Use same encoder structure
            sub_vecs = self.encode_sequence_block(sub_embs_pad, sub_lengths) # [num_subs, hid_dim]
            
            # DBSCAN Clustering
            # Move vectors to CPU for sklearn
            sub_vecs_np = sub_vecs.detach().cpu().numpy()
            
            if len(sub_vecs_np) > 0:
                clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(sub_vecs_np)
                labels = clustering.labels_
                
                # Calculate cluster centers
                unique_labels = set(labels)
                cluster_centers = []
                
                for label in unique_labels:
                    if label == -1: continue # Ignore noise
                    
                    indices = np.where(labels == label)[0]
                    center = np.mean(sub_vecs_np[indices], axis=0)
                    cluster_centers.append(center)
                
                if not cluster_centers:
                    # Fallback if no clusters found
                    C_nearest = S_whole[i]
                else:
                    cluster_centers = torch.tensor(np.array(cluster_centers), device=gol.device, dtype=torch.float)
                    
                    # Find nearest cluster center to S_whole
                    # S_whole[i]: [hid_dim]
                    # cluster_centers: [num_clusters, hid_dim]
                    dists = torch.norm(cluster_centers - S_whole[i].unsqueeze(0), dim=1)
                    min_idx = torch.argmin(dists)
                    C_nearest = cluster_centers[min_idx]
            else:
                 # Should not happen if len(subsequences) > 0
                 C_nearest = S_whole[i]

            # Calculate S_final
            # Formula: S_final = (1 - w) * S_whole + w * C_nearest
            S_final_i = (1 - guidance_w) * S_whole[i] + guidance_w * C_nearest
            S_final_list.append(S_final_i)
            
        S_final = torch.stack(S_final_list, dim=0)
        return S_final

    '''
    LocaGenerator: (C) Attention-based Location Archetype Generation Module in DiffDGMN
        Input: Sequence Encoding S_u, Node Geographical Representation R_V
        Return: Location Archetype Vector hat_L_u
    '''
    def LocaGenerator(self, POI_embs, seqs, S_u):
        # Node Geographical Representation R_V
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

        # output = output + Q
        output = output.squeeze(1)
        output = self.geo_attn_layernorm(output)

        hat_L_u = self.geo_PFFN(output)   # PFNN
        return hat_L_u, R_V

    '''
    DiffGenerator: (D) Diffusion-based User Preference Sampling Module in DiffDGMN
        Input: Location Archetype Vector hat_L_u, Sequence Encoding S_u as a context-aware condition embedding
        Return: A Pure (noise-free) Location Archetype Vector L_u
    '''
    def DiffGenerator(self, hat_L_u, S_u, seq_mean, target=None):
        local_embs = hat_L_u
        condition_embs = S_u.detach()

        # Reverse-time discrete interpolation process.
        L_u = self.SDEdiff.ReverseSDE_gener(local_embs, condition_embs, gol.conf['T'], x_end=seq_mean)

        loss_div = None
        if target is not None: # training phase
            t_sampled = np.random.randint(1, self.step_num) / self.step_num

            # Forward interpolation process: x_t = (1 - t) x_0 + t x_1
            perturbed_data = self.SDEdiff.ForwardSDE_diff(target, seq_mean, t_sampled)

            # Reconstruct x_0 = hat_L_u from the interpolated state.
            pred_x0 = self.SDEdiff.Est_score(perturbed_data, condition_embs, t_sampled)
            loss_div = torch.square(pred_x0 - target.detach()).mean()

        return L_u, loss_div



    '''Get cross-entropy recommendation loss and Fisher divergence loss'''
    def getTrainLoss(self, batch):
        usr, pos_lbl, exclude_mask, seqs, G_u, cur_time = batch

        R_v = self.poi_emb
        POI_embs = self.poi_emb
        if gol.conf['dropout']:
            POI_embs = self.dropout(POI_embs)

        seq_mean = self.get_sequence_mean(POI_embs, seqs)

        # Use new Sequence Encoder logic
        S_u = self.SeqGraphRep(POI_embs, seqs)
        
        hat_L_u, R_V = self.LocaGenerator(POI_embs, seqs, S_u)
        L_u, loss_div = self.DiffGenerator(hat_L_u, S_u, seq_mean, target=hat_L_u)

        Y_predscore = self.alpha * torch.matmul(S_u, R_v.t()) + torch.matmul(L_u, R_V.t())

        loss_rec = self.CEloss(Y_predscore, pos_lbl)
        return loss_rec, loss_div  


    def forward(self, seqs, G_u):
        R_v = self.poi_emb
        POI_embs = self.poi_emb

        seq_mean = self.get_sequence_mean(POI_embs, seqs)
        S_u = self.SeqGraphRep(POI_embs, seqs)
        hat_L_u, R_V = self.LocaGenerator(POI_embs, seqs, S_u)
        
        # --- Multiple Sampling ---
        sample_num = gol.conf.get('sample_num', 1)
        if sample_num > 1:
            L_u_list = []
            for _ in range(sample_num):
                sample, _ = self.DiffGenerator(hat_L_u, S_u, seq_mean)
                L_u_list.append(sample)
            L_u = torch.stack(L_u_list).mean(dim=0)
        else:
            L_u, _ = self.DiffGenerator(hat_L_u, S_u, seq_mean)
        # -------------------------

        '''
        Y_predscore = [batch_size, #POI]
        S_u         = [batch_size, hid_dim]
        R_v.T       = [hid_dim, #POI] 
        L_u         = [batch_size, hid_dim]
        R_V.T       = [hid_dim, #POI]
        '''

        Y_predscore = self.alpha * torch.matmul(S_u, R_v.t()) + torch.matmul(L_u, R_V.t()) 
        return Y_predscore
