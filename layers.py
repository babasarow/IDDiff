import gol
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, softmax

'''
PFFN: Point-wise Feed Forward Network
'''
class PFFN(nn.Module):
    def __init__(self, hid_size, dropout_rate):
        super(PFFN, self).__init__()
        self.conv1 = nn.Conv1d(hid_size, hid_size, kernel_size=1) 
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hid_size, hid_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs



'''
BiSeqGCN: Bi-directional Sequence Graph Convolution, is as a part of (A) Direction-aware Sequence Graph Multi-scale Representation Module (SeqGraphRep)
    Input: User-oriented POI Transition Graph G_u
    Return: Node representation of the user-oriented POI transition graph H_u
'''
class BiSeqGCN(MessagePassing):
    def __init__(self, hid_dim, flow="source_to_target"):
        super(BiSeqGCN, self).__init__(aggr='add', flow=flow)
        self.hid_dim = hid_dim
        self.alpha_src = nn.Linear(hid_dim, 1, bias=False)
        self.alpha_dst = nn.Linear(hid_dim, 1, bias=False)
        
        # attention_weight
        self.attention_weight = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        nn.init.xavier_uniform_(self.attention_weight.data)
        
        nn.init.xavier_uniform_(self.alpha_src.weight)
        nn.init.xavier_uniform_(self.alpha_dst.weight)
        self.act = nn.LeakyReLU()

    def forward(self, embs, G_u):
        POI_embs, delta_dis_embs, delta_time_embs = embs
        sess_idx   = G_u.x.squeeze()
        edge_index = G_u.edge_index
        edge_time  = G_u.edge_time
        edge_dist  = G_u.edge_dist
        
        x = POI_embs[sess_idx]
        edge_l = delta_dis_embs[edge_dist]
        edge_t = delta_time_embs[edge_time]
        all_edges = torch.cat((edge_index, edge_index[[1, 0]]), dim=-1)
        
        H_u = self.propagate(all_edges, x=x, edge_l=edge_l, edge_t=edge_t, edge_size=edge_index.size(1))
        return H_u

    def message(self, x_j, x_i, edge_index_j, edge_index_i, edge_l, edge_t, edge_size):
        attention_coefficients = torch.matmul(x_i[edge_size:] + edge_l + edge_t, self.attention_weight.t())
        
        src_attention = self.alpha_src(attention_coefficients[:edge_size]).squeeze(-1)
        dst_attention = self.alpha_dst(attention_coefficients[:edge_size]).squeeze(-1)
        
        # softmax on tot_attention
        tot_attention = torch.cat((src_attention, dst_attention), dim=0)
        attn_weight = softmax(tot_attention, edge_index_i)

        # attn_weight on neighbor node features
        updated_rep = x_j * attn_weight.unsqueeze(-1)
        return updated_rep

'''
SeqGraphEncoder: Encode BiSeqGCN, is as a part of (A) Direction-aware Sequence Graph Multi-scale Representation Module (SeqGraphRep)
'''
class SeqGraphEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(SeqGraphEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = BiSeqGCN(hid_dim)

    def encode(self, embs, G_u):
        return self.encoder(embs, G_u)



'''
DisDyGCN: Distance-based Dynamic Graph Convolution, is as a part of (B) Global-based Distance Graph Geographical Representation Module (DisGraphRep)
    Input: Global-based POI Distance Graph G_D
    Return: Updated Node Information h
'''
class DisDyGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, dist_embed_dim=64):
        super(DisDyGCN, self).__init__(aggr='add')
        self._cached_edge = None
        self.linear = nn.Linear(in_channels, out_channels)
        nn.init.xavier_uniform_(self.linear.weight)
        
        # dynamic mechanism on diatance
        self.dist_transform = nn.Sequential(
            nn.Linear(1, dist_embed_dim),  
            nn.ReLU(),
            nn.Linear(dist_embed_dim, out_channels)  
            )
        
        nn.init.xavier_uniform_(self.dist_transform[0].weight)
        nn.init.xavier_uniform_(self.dist_transform[2].weight)

    def forward(self, x, G_D: Data):
        if self._cached_edge is None:
            self._cached_edge = gcn_norm(G_D.edge_index, add_self_loops=False)
        edge_index, norm_weight = self._cached_edge
        x = self.linear(x)
        h = self.propagate(edge_index, x=x, norm=norm_weight, dist_vec=G_D.edge_attr)
        return h
    
    def message(self, x_j, norm, dist_vec):
        dist_weight = self.dist_transform(dist_vec.unsqueeze(-1))
        message_trans = norm.unsqueeze(-1) * x_j * dist_weight
        return message_trans

'''
DisGraphRep: (B) Global-based Distance Graph Geographical Representation Module in DiffDGMN
    Input: Global-based POI Distance Graph G_D
    Return: Node Geographical Representation R_V
'''
class DisGraphRep(nn.Module):
    def __init__(self, n_poi, hid_dim, G_D: Data):
        super(DisGraphRep, self).__init__()
        self.n_poi, self.hid_dim = n_poi, hid_dim
        self.GCN_layer = gol.conf['num_layer']

        # aggregating own features: 
        edge_index, _ = add_self_loops(G_D.edge_index)  
        dist_vec = torch.cat([G_D.edge_attr, torch.zeros((n_poi,)).to(gol.device)])
        # a_{i,j}^D: 
        dis_edgeweight = torch.exp(-(dist_vec ** 2)) 
        self.G_D = Data(edge_index=edge_index, edge_attr=dis_edgeweight)

        self.act = nn.LeakyReLU()
        self.DisDyGCN = nn.ModuleList()
        for _ in range(self.GCN_layer):
            self.DisDyGCN.append(DisDyGCN(self.hid_dim, self.hid_dim))

    def encode(self, poi_embs):
        layer_embs = poi_embs
        geo_embs = [layer_embs]
        
        for conv in self.DisDyGCN:
            layer_embs = conv(layer_embs, self.G_D) 
            layer_embs = self.act(layer_embs)
            geo_embs.append(layer_embs)

        R_V = torch.stack(geo_embs, dim=1).mean(1)
        return R_V  



'''
SDE_Diffusion: keep the original symbol name, but implement the user-specified
discrete interpolation diffusion to minimize downstream code changes.
'''
class SDE_Diffusion(nn.Module):
    def __init__(self, hid_dim, beta_min, beta_max, dt):
        super(SDE_Diffusion, self).__init__()
        self.hid_dim = hid_dim
        self.beta_min, self.beta_max = beta_min, beta_max
        self.dt = dt
        self.reverse_steps = max(1, int(gol.conf.get('interp_steps', max(1, gol.conf.get('T', 1)))))

        self.time_proj = nn.Sequential(
                    nn.Linear(1, hid_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hid_dim, hid_dim)
                    )
        self.score_FC = nn.Sequential(
                    nn.Linear(3 * hid_dim, 2 * hid_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(2 * hid_dim, hid_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hid_dim, hid_dim)
                    )

        for module in list(self.time_proj) + list(self.score_FC):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _time_embedding(self, t, batch_size, device, dtype):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        if t.dim() == 0:
            t = t.expand(batch_size)
        t = t.reshape(batch_size, 1)
        return self.time_proj(t)

    def Est_score(self, x, condition, t):
        t_emb = self._time_embedding(t, x.size(0), x.device, x.dtype)
        return self.score_FC(torch.cat((x, condition, t_emb), dim=-1))

    def ForwardSDE_diff(self, x0, x1, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x0.device, dtype=x0.dtype)
        t = t.to(device=x0.device, dtype=x0.dtype)
        if t.dim() == 0:
            t = t.expand(x0.size(0))
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        return (1.0 - t) * x0 + t * x1

    def ReverseSDE_gener(self, x_start, condition, T, x_end=None):
        total_steps = max(1, int(T) if T is not None else self.reverse_steps)
        total_steps = max(total_steps, self.reverse_steps)
        current = x_start
        if x_end is None:
            x_end = x_start

        for s in range(total_steps, 0, -1):
            t_cur = float(s) / float(total_steps)
            t_prev = float(s - 1) / float(total_steps)
            pred_x0 = self.Est_score(current, condition, t_cur)
            current = current - self.ForwardSDE_diff(pred_x0, x_end, t_cur) + self.ForwardSDE_diff(pred_x0, x_end, t_prev)

        return current
