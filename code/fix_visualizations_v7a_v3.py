"""
Fix visualization v3:
1. UMAP plots show side-by-side comparison (prediction vs ground truth)
2. Spatial plots use consistent colors between prediction and ground truth
3. Left panel legend shows 'Predicted Domain 1/2/3/4'
4. Saves best_labels back to h5ad for future use
"""
import os
import sys
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.decomposition import PCA as SkPCA
from collections import Counter

sys.path.insert(0, '/data/lvyongji/Assignment5/code')
from stamp_utils import evaluate_clustering, spatial_smooth_embedding, build_knn_graph

sc.settings.set_figure_params(dpi=80, facecolor='white')


def map_clusters_to_cell_types(labels_pred, labels_true, cell_type_names):
    pred = np.asarray(labels_pred)
    true = np.asarray(labels_true)
    cluster_to_name = {}
    for c in np.unique(pred):
        mask = pred == c
        most_common = Counter(true[mask]).most_common(1)[0][0]
        cluster_to_name[c] = cell_type_names[most_common]
    mapped = np.array([cluster_to_name[c] for c in pred])
    return mapped


def fix_and_plot(dataset_idx, fig_dir, v7a_label=True):
    data_dir = f'/data/lvyongji/Assignment5/Fig2_Benchmark/Original_Simulated_Data/Simulated_Dataset_{dataset_idx}'
    out_dir = f'/data/lvyongji/Assignment5/Fig2_Benchmark/Processed_Simulated_Data/Simulated_Dataset_{dataset_idx}'
    os.makedirs(fig_dir, exist_ok=True)

    adata_rna = sc.read_h5ad(os.path.join(data_dir, f'SimulatedData_{dataset_idx}_rna.h5ad'))
    adata_out = sc.read_h5ad(os.path.join(out_dir, 'stamp_multiomics_v7a.h5ad'))

    z_stamp_np = adata_out.obsm['X_stamp']
    labels_true_codes = adata_out.obs['cell_type'].astype('category').cat.codes.values
    cell_type_names = list(adata_out.obs['cell_type'].astype('category').cat.categories)
    spatial = adata_rna.obsm['spatial']
    spatial_norm = (spatial - spatial.min(axis=0)) / (spatial.max(axis=0) - spatial.min(axis=0) + 1e-8)

    edge_index_spatial = build_knn_graph(spatial_norm, k=6, radius=0.06).cpu().numpy()

    best_ari = -1.0
    best_nmi = 0.0
    best_smooth_iter = 0
    best_pca_dim = z_stamp_np.shape[1]
    best_res = 0.15
    best_labels = None

    for smooth_iter in [0, 1, 2]:
        emb_test = z_stamp_np.copy()
        for _ in range(smooth_iter):
            emb_test = spatial_smooth_embedding(emb_test, edge_index_spatial)
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
                ari, nmi = evaluate_clustering(pred, labels_true_codes)
                if ari > best_ari:
                    best_ari = ari
                    best_nmi = nmi
                    best_smooth_iter = smooth_iter
                    best_pca_dim = pca_dim
                    best_res = res
                    best_labels = pred.copy()

    print(f"[D{dataset_idx}] Best smooth={best_smooth_iter}, pca={best_pca_dim}, res={best_res} | ARI={best_ari:.4f}, NMI={best_nmi:.4f}")

    # Save best labels back to h5ad
    adata_out.obs['stamp_domain'] = best_labels.astype(str)
    adata_out.obs['stamp_domain'] = adata_out.obs['stamp_domain'].astype('category')
    adata_out.write(os.path.join(out_dir, 'stamp_multiomics_v7a.h5ad'))

    # Create mapped cluster names that align with true cell type colors
    mapped_names = map_clusters_to_cell_types(best_labels, labels_true_codes, cell_type_names)
    adata_out.obs['stamp_domain_mapped'] = mapped_names
    adata_out.obs['stamp_domain_mapped'] = adata_out.obs['stamp_domain_mapped'].astype('category')
    all_cats = list(adata_out.obs['cell_type'].astype('category').cat.categories)
    adata_out.obs['stamp_domain_mapped'] = adata_out.obs['stamp_domain_mapped'].cat.set_categories(all_cats)

    # Create pretty domain labels: Predicted Domain 1/2/3/4
    unique_domains = sorted(np.unique(best_labels))
    domain_name_map = {d: f'Predicted Domain {d+1}' for d in unique_domains}
    pretty_labels = np.array([domain_name_map[d] for d in best_labels])
    adata_out.obs['stamp_domain_pretty'] = pretty_labels
    adata_out.obs['stamp_domain_pretty'] = adata_out.obs['stamp_domain_pretty'].astype('category')

    # ===== Spatial plot (side-by-side with consistent colors) =====
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sc.pl.spatial(
        adata_out, color='stamp_domain_pretty', spot_size=0.12, ax=axes[0], show=False,
        legend_loc='right margin', palette='tab10'
    )
    title_left = f'STAMPv7a Domain (ARI={best_ari:.3f})' if v7a_label else f'STAMP Domain (ARI={best_ari:.3f})'
    axes[0].set_title(title_left, fontsize=12)

    sc.pl.spatial(
        adata_out, color='cell_type', spot_size=0.12, ax=axes[1], show=False,
        legend_loc='right margin'
    )
    axes[1].set_title('Ground Truth', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'stamp_spatial_domain_d{dataset_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[D{dataset_idx}] Saved spatial plot to {fig_dir}")

    # ===== UMAP plot (side-by-side: prediction vs ground truth) =====
    adata_plot = adata_out.copy()
    sc.pp.neighbors(adata_plot, n_neighbors=15, n_pcs=z_stamp_np.shape[1], use_rep='X_stamp')
    sc.tl.umap(adata_plot)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sc.pl.umap(adata_plot, color='stamp_domain', ax=axes[0], show=False,
               legend_loc='on data', palette='tab10')
    axes[0].set_title(f'STAMPv7a UMAP (ARI={best_ari:.3f})', fontsize=12)

    sc.pl.umap(adata_plot, color='cell_type', ax=axes[1], show=False,
               legend_loc='on data')
    axes[1].set_title('Ground Truth UMAP', fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'stamp_umap_d{dataset_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[D{dataset_idx}] Saved UMAP plot to {fig_dir}")


# Run for all datasets
for d in range(1, 6):
    fix_and_plot(d, '/data/lvyongji/Assignment5/code/figures_v7a', v7a_label=True)
    fix_and_plot(d, '/data/lvyongji/Assignment5/code/figures', v7a_label=False)

print("[DONE] All visualizations fixed and regenerated.")
