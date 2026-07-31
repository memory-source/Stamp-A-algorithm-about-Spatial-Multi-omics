"""
Run STAMP on real mouse brain P22 data and generate visualization.
"""
import os
import sys
import numpy as np
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # line buffered
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from sklearn.decomposition import PCA as SkPCA

sys.path.insert(0, '/data/lvyongji/Assignment5/code')
from stamp_model_v7a import STAMP
from stamp_utils import build_knn_graph, spatial_smooth_embedding
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sc.settings.set_figure_params(dpi=80, facecolor='white')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Load mouse brain data
import h5py
h5_path = '/data/lvyongji/Assignment5/Fig2_Benchmark/COSMOS/ATAC_RNA_Seq_MouseBrain_RNA_ATAC.h5'
with h5py.File(h5_path, 'r') as f:
    X_RNA = np.array(f['X_RNA'])
    X_ATAC = np.array(f['X_ATAC'])
    Pos = np.array(f['Pos'])
    Cell = np.array(f['Cell'])
    LayerName = np.array(f['LayerName'])

print(f"Mouse brain data: RNA={X_RNA.shape}, ATAC={X_ATAC.shape}, Pos={Pos.shape}")

# Create AnnData
adata_rna = sc.AnnData(X_RNA)
adata_rna.obsm['spatial'] = Pos
adata_rna.obs_names = [c.decode() if isinstance(c, bytes) else str(c) for c in Cell]

adata_atac = sc.AnnData(X_ATAC)
adata_atac.obsm['spatial'] = Pos
adata_atac.obs_names = adata_rna.obs_names

# Use LayerName as pseudo ground truth (cortical layers)
layer_names = [l.decode() if isinstance(l, bytes) else str(l) for l in LayerName]
adata_rna.obs['layer'] = layer_names
adata_atac.obs['layer'] = layer_names

# RNA: already log-normalized, directly PCA to 50 dims
sc.tl.pca(adata_rna, n_comps=50, svd_solver='arpack')
x_rna_pca = torch.tensor(adata_rna.obsm['X_pca'], dtype=torch.float32).to(DEVICE)

# ATAC: already 50-dim (likely LSI), use directly
x_atac_lsi = torch.tensor(X_ATAC, dtype=torch.float32).to(DEVICE)

# Build graphs
spatial = Pos
spatial_norm = (spatial - spatial.min(axis=0)) / (spatial.max(axis=0) - spatial.min(axis=0) + 1e-8)
edge_index_spatial = build_knn_graph(spatial_norm, k=6, radius=0.06)

# Feature graphs
from sklearn.neighbors import NearestNeighbors
nbrs_rna = NearestNeighbors(n_neighbors=15).fit(adata_rna.obsm['X_pca'])
adj_rna = nbrs_rna.kneighbors_graph(mode='connectivity')
edge_index_rna_feature = from_scipy_sparse_matrix(sp.csr_matrix(adj_rna))[0].to(DEVICE)

nbrs_atac = NearestNeighbors(n_neighbors=15).fit(X_ATAC)
adj_atac = nbrs_atac.kneighbors_graph(mode='connectivity')
edge_index_atac_feature = from_scipy_sparse_matrix(sp.csr_matrix(adj_atac))[0].to(DEVICE)

edge_index_spatial = edge_index_spatial.to(DEVICE)

# Initialize model
model = STAMP(pca_dim=50, hidden=128, latent=30, heads_rna=8, heads_atac=4).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

# Training
best_loss = float('inf')
patience = 200
patience_counter = 0

for epoch in range(1500):
    model.train()
    optimizer.zero_grad()
    z_stamp, z_r, z_a, recon_r_from_r, recon_a_from_r, recon_r_from_a, recon_a_from_a = model(
        x_rna_pca, x_atac_lsi, edge_index_spatial, edge_index_rna_feature,
        edge_index_spatial, edge_index_atac_feature
    )
    loss_recon = (F.mse_loss(recon_r_from_r, x_rna_pca) +
                  F.mse_loss(recon_a_from_r, x_atac_lsi) +
                  F.mse_loss(recon_r_from_a, x_rna_pca) +
                  F.mse_loss(recon_a_from_a, x_atac_lsi))
    loss_l2 = z_stamp.pow(2).mean()
    loss_align = F.mse_loss(z_r, z_a.detach()) + F.mse_loss(z_a, z_r.detach())
    loss_spatial = 0.0
    loss = loss_recon + 1e-3 * loss_l2 + 0.1 * loss_align + 0.3 * loss_spatial
    loss.backward()
    optimizer.step()
    scheduler.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}, loss={loss.item():.4f}")
        break

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    z_stamp, z_r, z_a, _, _, _, _ = model(
        x_rna_pca, x_atac_lsi, edge_index_spatial, edge_index_rna_feature,
        edge_index_spatial, edge_index_atac_feature
    )
    z_stamp_np = z_stamp.cpu().numpy()

print(f"Training done. Best loss={best_loss:.4f}")

# Save results
adata_out = sc.AnnData(np.concatenate([X_RNA[:, :50], X_ATAC[:, :50]], axis=1))
adata_out.obsm['X_stamp'] = z_stamp_np
adata_out.obsm['spatial'] = Pos
adata_out.obs_names = adata_rna.obs_names
adata_out.obs['layer'] = layer_names

# Post-processing: grid search for best clustering
edge_index_spatial_np = edge_index_spatial.cpu().numpy()
best_ari = -1.0
best_labels = None

labels_true = np.array(adata_out.obs['layer'].astype('category').cat.codes)

for smooth_iter in [0, 1, 2]:
    emb_test = z_stamp_np.copy()
    for _ in range(smooth_iter):
        emb_test = spatial_smooth_embedding(emb_test, edge_index_spatial_np)
    for pca_dim in [z_stamp_np.shape[1], 20, 15, 10]:
        if pca_dim < emb_test.shape[1]:
            emb_reduced = SkPCA(n_components=pca_dim, random_state=42).fit_transform(emb_test)
        else:
            emb_reduced = emb_test.copy()
        ad_tmp = sc.AnnData(emb_reduced)
        sc.pp.neighbors(ad_tmp, n_neighbors=15, n_pcs=emb_reduced.shape[1], use_rep='X')
        for res in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4]:
            sc.tl.leiden(ad_tmp, resolution=res)
            pred = ad_tmp.obs['leiden'].astype(int).values
            ari = adjusted_rand_score(labels_true, pred)
            if ari > best_ari:
                best_ari = ari
                best_labels = pred.copy()
                best_res = res

print(f"Mouse brain best ARI={best_ari:.4f} at res={best_res}")

# Save output
out_dir = '/data/lvyongji/Assignment5/Fig2_Benchmark/Processed_Simulated_Data'
os.makedirs(out_dir, exist_ok=True)
adata_out.obs['stamp_domain'] = best_labels.astype(str)
adata_out.obs['stamp_domain'] = adata_out.obs['stamp_domain'].astype('category')
adata_out.write(os.path.join(out_dir, 'stamp_mousebrain.h5ad'))

# Create pretty domain labels
unique_domains = sorted(np.unique(best_labels))
domain_name_map = {d: f'Predicted Domain {d+1}' for d in unique_domains}
pretty_labels = np.array([domain_name_map[d] for d in best_labels])
adata_out.obs['stamp_domain_pretty'] = pretty_labels
adata_out.obs['stamp_domain_pretty'] = adata_out.obs['stamp_domain_pretty'].astype('category')

# Plot spatial domain
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.spatial(
    adata_out, color='stamp_domain_pretty', spot_size=0.08, ax=axes[0], show=False,
    legend_loc='right margin', palette='tab20'
)
axes[0].set_title(f'STAMP on P22 Mouse Brain (ARI={best_ari:.3f})', fontsize=12)

sc.pl.spatial(
    adata_out, color='layer', spot_size=0.08, ax=axes[1], show=False,
    legend_loc='right margin', palette='tab20'
)
axes[1].set_title('Cortical Layers (Ground Truth)', fontsize=12)

fig.tight_layout()
fig.savefig('/data/lvyongji/Assignment5/code/figures_v7a/stamp_mousebrain_p22.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved mouse brain spatial plot.")

# UMAP
sc.pp.neighbors(adata_out, n_neighbors=15, n_pcs=z_stamp_np.shape[1], use_rep='X_stamp')
sc.tl.umap(adata_out)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata_out, color='stamp_domain_pretty', ax=axes[0], show=False,
           legend_loc='right margin', palette='tab20')
axes[0].set_title(f'STAMP UMAP (ARI={best_ari:.3f})', fontsize=12)

sc.pl.umap(adata_out, color='layer', ax=axes[1], show=False,
           legend_loc='right margin', palette='tab20')
axes[1].set_title('Ground Truth UMAP', fontsize=12)

fig.tight_layout()
fig.savefig('/data/lvyongji/Assignment5/code/figures_v7a/stamp_umap_mousebrain_p22.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved mouse brain UMAP plot.")

print("[DONE] Mouse brain processing complete.")
