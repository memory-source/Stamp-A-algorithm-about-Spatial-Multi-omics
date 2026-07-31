"""
STAMP: Dual-graph encoder with asymmetric cross-modal attention.
Spatial + feature kNN graph for each modality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class DualGATEncoder(nn.Module):
    """
    GAT encoder processing both spatial graph and feature-similarity graph.
    Two graphs share GAT weights; activations fused with learnable alpha.
    """
    def __init__(self, in_dim=50, hidden_dim=128, out_dim=30, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=0.0)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=0.0)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        self.proj = nn.Linear(hidden_dim * heads, out_dim)
        self.skip = nn.Linear(in_dim, out_dim)
        self.bn_out = nn.BatchNorm1d(out_dim)
        # Learnable fusion weight for spatial vs feature graph (init 0.85 favor spatial)
        self.graph_alpha = nn.Parameter(torch.tensor(0.85))

    def forward(self, x, edge_index_spatial, edge_index_feature=None):
        if edge_index_feature is None:
            h = F.elu(self.gat1(x, edge_index_spatial))
            h = self.bn1(h)
            h = F.elu(self.gat2(h, edge_index_spatial))
            h = self.bn2(h)
            z = self.proj(h) + self.skip(x)
            z = self.bn_out(z)
            return z

        # Layer 1: both graphs, then fuse
        h_s = self.gat1(x, edge_index_spatial)
        h_f = self.gat1(x, edge_index_feature)
        alpha = torch.sigmoid(self.graph_alpha)
        h = alpha * h_s + (1 - alpha) * h_f
        h = F.elu(h)
        h = self.bn1(h)

        # Layer 2: both graphs, then fuse
        h_s = self.gat2(h, edge_index_spatial)
        h_f = self.gat2(h, edge_index_feature)
        h = alpha * h_s + (1 - alpha) * h_f
        h = F.elu(h)
        h = self.bn2(h)

        z = self.proj(h) + self.skip(x)
        z = self.bn_out(z)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=30, hidden_dim=128, out_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.net(z)


class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.residual_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.residual_gate[0].bias.data.fill_(2.0)

    def forward(self, x, y):
        Q = self.q_proj(x)
        K = self.k_proj(y)
        V = self.v_proj(y)
        attn = torch.softmax(Q @ K.T * self.scale, dim=-1)
        out = attn @ V
        gate = self.residual_gate(torch.cat([x, out], dim=-1))
        return x + gate * out


class STAMP(nn.Module):
    def __init__(self, pca_dim=50, hidden=128, latent=30, heads_rna=8, heads_atac=4):
        super().__init__()
        self.rna_enc = DualGATEncoder(pca_dim, hidden, latent, heads=heads_rna)
        self.atac_enc = DualGATEncoder(pca_dim, hidden, latent, heads=heads_atac)
        self.cma_r_to_a = CrossModalAttention(latent)
        self.dec_rna_from_rna = Decoder(latent, hidden, pca_dim)
        self.dec_atac_from_rna = Decoder(latent, hidden, pca_dim)
        self.dec_rna_from_atac = Decoder(latent, hidden, pca_dim)
        self.dec_atac_from_atac = Decoder(latent, hidden, pca_dim)

    def forward(self, x_rna_pca, x_atac_pca, edge_index_rna_spatial, edge_index_rna_feature,
                edge_index_atac_spatial, edge_index_atac_feature):
        z_r_base = self.rna_enc(x_rna_pca, edge_index_rna_spatial, edge_index_rna_feature)
        z_a = self.atac_enc(x_atac_pca, edge_index_atac_spatial, edge_index_atac_feature)
        z_r = self.cma_r_to_a(z_r_base, z_a)
        z_stamp = torch.cat([z_r, z_a], dim=-1)
        recon_r_from_r = self.dec_rna_from_rna(z_r_base)
        recon_a_from_r = self.dec_atac_from_rna(z_r_base)
        recon_r_from_a = self.dec_rna_from_atac(z_a)
        recon_a_from_a = self.dec_atac_from_atac(z_a)
        return z_stamp, z_r, z_a, recon_r_from_r, recon_a_from_r, recon_r_from_a, recon_a_from_a


def spatial_aware_infonce(z_a, z_b, spatial_neighbors=None, temperature=0.1):
    n = z_a.size(0)
    sim = F.cosine_similarity(z_a.unsqueeze(1), z_b.unsqueeze(0), dim=-1) / temperature
    pos_mask = torch.zeros(n, n, device=z_a.device)
    pos_mask[torch.arange(n), torch.arange(n)] = 1.0
    if spatial_neighbors is not None:
        for i in range(n):
            neighbors = spatial_neighbors[i]
            if len(neighbors) > 0:
                pos_mask[i, neighbors] = 0.5
    pos_sim = sim * pos_mask
    numerator = torch.logsumexp(pos_sim + (1 - pos_mask.bool().float()) * (-1e9), dim=1)
    denominator = torch.logsumexp(sim, dim=1)
    loss = -numerator + denominator
    return loss.mean()
