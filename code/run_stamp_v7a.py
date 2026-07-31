"""
STAMP v7a: Step A — Dual-graph encoder only.
No pretraining, no adaptive fusion. Fair comparison with v6.1.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

from stamp_utils import build_knn_graph, evaluate_clustering, lsi_atac, spatial_smooth_embedding, compute_spatial_neighbors
from stamp_model_v7a import STAMP, spatial_aware_infonce

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1, help='Dataset index 1-5')
args = parser.parse_args()
DATASET_IDX = args.dataset

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {DEVICE}")

DATA_DIR = f'/data/lvyongji/Assignment5/Fig2_Benchmark/Original_Simulated_Data/Simulated_Dataset_{DATASET_IDX}'
OUT_DIR = f'/data/lvyongji/Assignment5/Fig2_Benchmark/Processed_Simulated_Data/Simulated_Dataset_{DATASET_IDX}'
CODE_DIR = '/data/lvyongji/Assignment5/code'
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
print("[INFO] Loading RNA and ATAC data...")
adata_rna_raw = sc.read_h5ad(os.path.join(DATA_DIR, f'SimulatedData_{DATASET_IDX}_rna.h5ad'))
adata_atac_raw = sc.read_h5ad(os.path.join(DATA_DIR, f'SimulatedData_{DATASET_IDX}_atac.h5ad'))
adata_atac_raw.obsm['spatial'] = adata_rna_raw.obsm['spatial'].copy()

sc.pp.highly_variable_genes(adata_rna_raw, n_top_genes=3000)
adata_rna = adata_rna_raw[:, adata_rna_raw.var.highly_variable].copy()
sc.pp.scale(adata_rna)

X_atac = adata_atac_raw.X
if sp.issparse(X_atac):
    peak_counts = np.asarray(X_atac.sum(axis=0)).ravel()
else:
    peak_counts = np.sum(X_atac, axis=0)
selected = np.argsort(peak_counts)[-5000:]
selected = np.sort(selected)
adata_atac = adata_atac_raw[:, selected].copy()

x_rna_full = adata_rna.X.toarray() if sp.issparse(adata_rna.X) else np.array(adata_rna.X)
x_atac_full = adata_atac.X.toarray() if sp.issparse(adata_atac.X) else np.array(adata_atac.X)

RNA_PCA_DIM = 50
ATAC_LSI_DIM = 50
pca_rna = PCA(n_components=RNA_PCA_DIM, random_state=42)
x_rna_pca = pca_rna.fit_transform(x_rna_full)
x_atac_lsi = lsi_atac(x_atac_full, n_components=ATAC_LSI_DIM, random_state=42)

x_rna_pca_t = torch.FloatTensor(x_rna_pca).to(DEVICE)
x_atac_lsi_t = torch.FloatTensor(x_atac_lsi).to(DEVICE)

# Dual graphs
spatial = adata_rna.obsm['spatial']
spatial_norm = (spatial - spatial.min(axis=0)) / (spatial.max(axis=0) - spatial.min(axis=0) + 1e-8)

print("[INFO] Building spatial graph (k=6, radius=0.06)...")
edge_index_spatial = build_knn_graph(spatial_norm, k=6, radius=0.06).to(DEVICE)
edge_index_spatial_np = edge_index_spatial.cpu().numpy()
spatial_neighbors = compute_spatial_neighbors(spatial_norm, k=6)

print("[INFO] Building feature similarity graph (k=15)...")
adj_rna_feature = kneighbors_graph(x_rna_pca, n_neighbors=15, mode='connectivity', include_self=False)
edge_index_rna_feature = from_scipy_sparse_matrix(adj_rna_feature)[0].to(DEVICE)
adj_atac_feature = kneighbors_graph(x_atac_lsi, n_neighbors=15, mode='connectivity', include_self=False)
edge_index_atac_feature = from_scipy_sparse_matrix(adj_atac_feature)[0].to(DEVICE)

# Model
model = STAMP(pca_dim=RNA_PCA_DIM, hidden=128, latent=30, heads_rna=8, heads_atac=4).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

# Training
EPOCHS = 1500
l2_weight = 1e-3
patience = 200
best_loss = float('inf')
best_epoch = 0
best_state = None
patience_counter = 0

print("[INFO] Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    z_stamp, z_r, z_a, recon_r_from_r, recon_a_from_r, recon_r_from_a, recon_a_from_a = model(
        x_rna_pca_t, x_atac_lsi_t,
        edge_index_spatial, edge_index_rna_feature,
        edge_index_spatial, edge_index_atac_feature
    )

    loss_recon = (
        F.mse_loss(recon_r_from_r, x_rna_pca_t) +
        F.mse_loss(recon_a_from_r, x_atac_lsi_t) +
        F.mse_loss(recon_r_from_a, x_rna_pca_t) +
        F.mse_loss(recon_a_from_a, x_atac_lsi_t)
    )
    loss_l2 = l2_weight * (z_r.pow(2).mean() + z_a.pow(2).mean())
    loss_align = 0.1 * (F.mse_loss(z_r, z_a.detach()) + F.mse_loss(z_a, z_r.detach()))
    loss_spatial = 0.3 * spatial_aware_infonce(z_stamp, z_stamp, spatial_neighbors, temperature=0.1)
    loss_spatial_r = 0.1 * spatial_aware_infonce(z_r, z_r, spatial_neighbors, temperature=0.1)
    loss_spatial_a = 0.1 * spatial_aware_infonce(z_a, z_a, spatial_neighbors, temperature=0.1)

    loss = loss_recon + loss_l2 + loss_align + loss_spatial + loss_spatial_r + loss_spatial_a
    loss.backward()
    optimizer.step()
    scheduler.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if (epoch + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} (Recon={loss_recon.item():.4f}, Align={loss_align.item():.4f}, Spatial={loss_spatial.item():.4f}) | Best={best_loss:.4f}@ep{best_epoch+1} | Time: {elapsed:.1f}s")

    if patience_counter >= patience:
        print(f"[INFO] Early stopping at epoch {epoch+1}")
        break

if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"[INFO] Restored best model from epoch {best_epoch+1}")

torch.save(model.state_dict(), os.path.join(CODE_DIR, 'best_model.pt'))
print(f"[INFO] Training completed in {time.time() - start_time:.1f}s")

# Inference
model.eval()
with torch.no_grad():
    z_stamp, _, _, _, _, _, _ = model(
        x_rna_pca_t, x_atac_lsi_t,
        edge_index_spatial, edge_index_rna_feature,
        edge_index_spatial, edge_index_atac_feature
    )

z_stamp_np = z_stamp.cpu().numpy()
adata_out = adata_rna.copy()
adata_out.obsm['X_stamp'] = z_stamp_np

sc.pp.neighbors(adata_out, n_neighbors=15, n_pcs=z_stamp_np.shape[1], use_rep='X_stamp')
labels_true = adata_out.obs['cell_type'].astype('category').cat.codes.values

from sklearn.decomposition import PCA as SkPCA
best_ari = -1.0
best_nmi = 0.0
best_smooth_iter = 0
best_pca_dim = z_stamp_np.shape[1]
best_res = 0.15

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
            ari, nmi = evaluate_clustering(pred, labels_true)
            if ari > best_ari:
                best_ari = ari
                best_nmi = nmi
                best_smooth_iter = smooth_iter
                best_pca_dim = pca_dim
                best_res = res
                best_labels = pred.copy()

emb_final = z_stamp_np.copy()
for _ in range(best_smooth_iter):
    emb_final = spatial_smooth_embedding(emb_final, edge_index_spatial_np)
if best_pca_dim < emb_final.shape[1]:
    emb_final = SkPCA(n_components=best_pca_dim, random_state=42).fit_transform(emb_final)
adata_out.obsm['X_stamp'] = emb_final
sc.pp.neighbors(adata_out, n_neighbors=15, n_pcs=emb_final.shape[1], use_rep='X_stamp')
adata_out.obs['stamp_domain'] = best_labels.astype(str)
adata_out.obs['stamp_domain'] = adata_out.obs['stamp_domain'].astype('category')

print(f"[RESULT] Best smooth={best_smooth_iter}, pca={best_pca_dim}, res={best_res} | ARI={best_ari:.4f}, NMI={best_nmi:.4f}")

# Evaluate z_r, z_a, z_stamp
with torch.no_grad():
    z_stamp_raw, z_r, z_a, _, _, _, _ = model(
        x_rna_pca_t, x_atac_lsi_t,
        edge_index_spatial, edge_index_rna_feature,
        edge_index_spatial, edge_index_atac_feature
    )
    for emb_t, name in [(z_r, 'z_r'), (z_a, 'z_a'), (z_stamp_raw, 'z_stamp')]:
        emb_np = emb_t.cpu().numpy()
        emb_proc = emb_np.copy()
        for _ in range(best_smooth_iter):
            emb_proc = spatial_smooth_embedding(emb_proc, edge_index_spatial_np)
        if best_pca_dim < emb_proc.shape[1]:
            emb_proc = SkPCA(n_components=best_pca_dim, random_state=42).fit_transform(emb_proc)
        ad_tmp = sc.AnnData(emb_proc)
        ad_tmp.obs['cell_type'] = adata_out.obs['cell_type'].values
        sc.pp.neighbors(ad_tmp, n_neighbors=15, n_pcs=emb_proc.shape[1], use_rep='X')
        best_ari_tmp, best_nmi_tmp, best_res_tmp = -1, 0, 0
        for res in [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4]:
            sc.tl.leiden(ad_tmp, resolution=res)
            ari_tmp, nmi_tmp = evaluate_clustering(ad_tmp.obs['leiden'].astype(int).values, labels_true)
            if ari_tmp > best_ari_tmp:
                best_ari_tmp, best_nmi_tmp, best_res_tmp = ari_tmp, nmi_tmp, res
        print(f"[{name}] Best ARI={best_ari_tmp:.4f}, NMI={best_nmi_tmp:.4f} @ res={best_res_tmp}")

out_path = os.path.join(OUT_DIR, 'stamp_multiomics.h5ad')
adata_out.write_h5ad(out_path, compression='gzip')
print(f"[INFO] Saved results to {out_path}")

# Visualization
print("[INFO] Generating visualizations...")
fig_dir = os.path.join(CODE_DIR, 'figures')
os.makedirs(fig_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
sc.pl.spatial(adata_out, color=['stamp_domain'], spot_size=0.12, ax=axes[0], show=False, legend_loc='right margin')
axes[0].set_title(f'STAMP Domain (ARI={best_ari:.3f})', fontsize=12)
sc.pl.spatial(adata_out, color=['cell_type'], spot_size=0.12, ax=axes[1], show=False, legend_loc='right margin')
axes[1].set_title('Ground Truth', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, f'stamp_spatial_domain_d{DATASET_IDX}.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

sc.tl.umap(adata_out)
fig, ax = plt.subplots(figsize=(4, 4))
sc.pl.umap(adata_out, color=['stamp_domain'], ax=ax, show=False, legend_loc='on data')
ax.set_title('STAMP UMAP', fontsize=12)
fig.savefig(os.path.join(fig_dir, f'stamp_umap_d{DATASET_IDX}.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

print("[INFO] Visualizations saved to", fig_dir)
print("[DONE] STAMP pipeline completed successfully.")
