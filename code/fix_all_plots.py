"""
Fix all visualization issues:
1. Mouse brain: empty plot due to spot_size too small for coordinate scale
2. Spatial domain plots: legend overlap with y-axis label
3. Color consistency between predicted domains and ground truth
"""
import os
import sys
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from collections import Counter
import h5py

sys.path.insert(0, '/data/lvyongji/Assignment5/code')
from stamp_utils import evaluate_clustering

sc.settings.set_figure_params(dpi=80, facecolor='white')


def map_clusters_to_cell_types(labels_pred, labels_true, cell_type_names):
    """Map each predicted cluster to the most common true cell type name."""
    pred = np.asarray(labels_pred)
    true = np.asarray(labels_true)
    cluster_to_name = {}
    for c in np.unique(pred):
        mask = pred == c
        most_common = Counter(true[mask]).most_common(1)[0][0]
        cluster_to_name[c] = cell_type_names[most_common]
    mapped = np.array([cluster_to_name[c] for c in pred])
    return mapped


def get_color_mapping(adata, color_col, palette_name='tab10'):
    """Get a color mapping dict for a categorical column."""
    cats = list(adata.obs[color_col].astype('category').cat.categories)
    n_cats = len(cats)
    if palette_name == 'tab10':
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_cats]
    elif palette_name == 'tab20':
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_cats]
    elif palette_name == 'Set1':
        colors = plt.cm.Set1(np.linspace(0, 1, 9))[:n_cats]
    elif palette_name == 'Set2':
        colors = plt.cm.Set2(np.linspace(0, 1, 8))[:n_cats]
    else:
        colors = plt.cm.get_cmap(palette_name)(np.linspace(0, 1, n_cats))
    return {cat: colors[i] for i, cat in enumerate(cats)}


def plot_spatial_consistent(adata, pred_col, gt_col, title_left, title_right, 
                            spot_size, figsize, output_path, pred_palette='tab10'):
    """
    Plot spatial domain with consistent colors between prediction and ground truth.
    Uses manual scatter plot to avoid scanpy legend overlap issues.
    """
    coords = adata.obsm['spatial']
    
    # Get ground truth categories and colors
    gt_cats = list(adata.obs[gt_col].astype('category').cat.categories)
    n_gt = len(gt_cats)
    
    # Use tab20 for more colors if needed
    if n_gt <= 10:
        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif n_gt <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        # Extend with additional colors
        extra = plt.cm.Set3(np.linspace(0, 1, 12))
        base_colors = np.vstack([base_colors, extra])
    
    gt_colors = {cat: base_colors[i] for i, cat in enumerate(gt_cats)}
    
    # For predicted, we need to map clusters to GT names for color consistency
    # But show "Predicted Domain X" in legend
    pred_vals = adata.obs[pred_col].values
    pred_cats = list(adata.obs[pred_col].astype('category').cat.categories)
    
    # Create mapping from predicted cluster -> GT color
    # pred_col should already be mapped to GT names for color purposes
    # But we want legend to show "Predicted Domain X"
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left panel: predicted domains
    for i, cat in enumerate(pred_cats):
        mask = pred_vals == cat
        # Extract the GT-mapped color for this predicted cluster
        # The pred_col values should be the mapped GT names
        color = gt_colors.get(cat, base_colors[i % len(base_colors)])
        axes[0].scatter(coords[mask, 0], coords[mask, 1], 
                       c=[color], s=spot_size, label=cat, edgecolors='none')
    
    axes[0].set_title(title_left, fontsize=12)
    axes[0].set_xlabel('spatial1', fontsize=10)
    axes[0].set_ylabel('spatial2', fontsize=10)
    axes[0].set_aspect('equal')
    axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=8, frameon=False, title='')
    
    # Right panel: ground truth
    gt_vals = adata.obs[gt_col].values
    for i, cat in enumerate(gt_cats):
        mask = gt_vals == cat
        color = gt_colors[cat]
        axes[1].scatter(coords[mask, 0], coords[mask, 1], 
                       c=[color], s=spot_size, label=cat, edgecolors='none')
    
    axes[1].set_title(title_right, fontsize=12)
    axes[1].set_xlabel('spatial1', fontsize=10)
    axes[1].set_ylabel('spatial2', fontsize=10)
    axes[1].set_aspect('equal')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=8, frameon=False, title='')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def fix_simulated_plot(dataset_idx, fig_dir):
    """Regenerate spatial domain plot for simulated dataset with fixes."""
    data_dir = f'/data/lvyongji/Assignment5/Fig2_Benchmark/Original_Simulated_Data/Simulated_Dataset_{dataset_idx}'
    out_dir = f'/data/lvyongji/Assignment5/Fig2_Benchmark/Processed_Simulated_Data/Simulated_Dataset_{dataset_idx}'
    os.makedirs(fig_dir, exist_ok=True)

    adata_rna = sc.read_h5ad(os.path.join(data_dir, f'SimulatedData_{dataset_idx}_rna.h5ad'))
    adata_out = sc.read_h5ad(os.path.join(out_dir, 'stamp_multiomics_v7a.h5ad'))

    z_stamp_np = adata_out.obsm['X_stamp']
    labels_true_codes = adata_out.obs['cell_type'].astype('category').cat.codes.values
    cell_type_names = list(adata_out.obs['cell_type'].astype('category').cat.categories)

    best_labels = adata_out.obs['stamp_domain'].astype(int).values
    ari, nmi = evaluate_clustering(best_labels, labels_true_codes)
    print(f"[D{dataset_idx}] ARI={ari:.4f}, NMI={nmi:.4f}")

    # Map predicted clusters to GT cell type names for color consistency
    mapped_names = map_clusters_to_cell_types(best_labels, labels_true_codes, cell_type_names)
    
    # Create pretty domain labels for display
    unique_domains = sorted(np.unique(best_labels))
    domain_name_map = {d: f'Predicted Domain {d+1}' for d in unique_domains}
    pretty_labels = np.array([domain_name_map[d] for d in best_labels])
    
    # Build a mapping from pretty label -> GT name for color lookup
    pretty_to_gt = {}
    for d in unique_domains:
        mask = best_labels == d
        pretty_to_gt[domain_name_map[d]] = mapped_names[mask][0]
    
    # Set up adata with consistent categories for both panels
    adata_out.obs['stamp_domain_mapped'] = mapped_names
    adata_out.obs['stamp_domain_mapped'] = adata_out.obs['stamp_domain_mapped'].astype('category')
    all_cats = list(adata_out.obs['cell_type'].astype('category').cat.categories)
    adata_out.obs['stamp_domain_mapped'] = adata_out.obs['stamp_domain_mapped'].cat.set_categories(all_cats)
    
    # For the pretty labels, we need to preserve the mapping to GT colors
    # We'll create a custom color dict
    n_cats = len(all_cats)
    if n_cats <= 10:
        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    gt_color_dict = {cat: base_colors[i] for i, cat in enumerate(all_cats)}
    
    # Now plot manually to avoid legend overlap
    coords = adata_out.obsm['spatial']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: predicted domains with pretty labels but GT colors
    for i, d in enumerate(unique_domains):
        mask = best_labels == d
        pretty_name = domain_name_map[d]
        gt_name = pretty_to_gt[pretty_name]
        color = gt_color_dict[gt_name]
        axes[0].scatter(coords[mask, 0], coords[mask, 1], 
                       c=[color], s=80, label=pretty_name, edgecolors='none')
    
    axes[0].set_title(f'STAMP Domain (ARI={ari:.3f})', fontsize=12)
    axes[0].set_xlabel('spatial1', fontsize=10)
    axes[0].set_ylabel('spatial2', fontsize=10)
    axes[0].set_aspect('equal')
    axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=9, frameon=False)
    
    # Right: ground truth
    gt_vals = adata_out.obs['cell_type'].values
    for i, cat in enumerate(all_cats):
        mask = gt_vals == cat
        color = gt_color_dict[cat]
        axes[1].scatter(coords[mask, 0], coords[mask, 1], 
                       c=[color], s=80, label=cat, edgecolors='none')
    
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].set_xlabel('spatial1', fontsize=10)
    axes[1].set_ylabel('spatial2', fontsize=10)
    axes[1].set_aspect('equal')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=9, frameon=False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'stamp_spatial_domain_d{dataset_idx}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[D{dataset_idx}] Saved spatial plot to {fig_dir}")
    
    # UMAP plot
    adata_plot = adata_out.copy()
    sc.pp.neighbors(adata_plot, n_neighbors=15, n_pcs=z_stamp_np.shape[1], use_rep='X_stamp')
    sc.tl.umap(adata_plot)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left UMAP: predicted with pretty labels
    for i, d in enumerate(unique_domains):
        mask = best_labels == d
        pretty_name = domain_name_map[d]
        gt_name = pretty_to_gt[pretty_name]
        color = gt_color_dict[gt_name]
        # Need UMAP coords
        umap_coords = adata_plot.obsm['X_umap']
        axes[0].scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                       c=[color], s=20, label=pretty_name, edgecolors='none', alpha=0.7)
    
    axes[0].set_title(f'STAMP UMAP (ARI={ari:.3f})', fontsize=12)
    axes[0].set_xlabel('UMAP1', fontsize=10)
    axes[0].set_ylabel('UMAP2', fontsize=10)
    axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=8, frameon=False)
    
    # Right UMAP: ground truth
    for i, cat in enumerate(all_cats):
        mask = gt_vals == cat
        color = gt_color_dict[cat]
        axes[1].scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                       c=[color], s=20, label=cat, edgecolors='none', alpha=0.7)
    
    axes[1].set_title('Ground Truth UMAP', fontsize=12)
    axes[1].set_xlabel('UMAP1', fontsize=10)
    axes[1].set_ylabel('UMAP2', fontsize=10)
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=8, frameon=False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'stamp_umap_d{dataset_idx}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[D{dataset_idx}] Saved UMAP plot to {fig_dir}")


def fix_mousebrain_plot(fig_dir):
    """Generate mouse brain spatial plot with correct spot size."""
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load data
    h5_path = '/data/lvyongji/Assignment5/Fig2_Benchmark/COSMOS/ATAC_RNA_Seq_MouseBrain_RNA_ATAC.h5'
    with h5py.File(h5_path, 'r') as f:
        X_RNA = np.array(f['X_RNA'])
        X_ATAC = np.array(f['X_ATAC'])
        Pos = np.array(f['Pos'])
        Cell = np.array(f['Cell'])
        LayerName = np.array(f['LayerName'])
    
    layer_names = [l.decode() if isinstance(l, bytes) else str(l) for l in LayerName]
    
    # Check if we have saved results
    out_dir = '/data/lvyongji/Assignment5/Fig2_Benchmark/Processed_Simulated_Data'
    # Try v7a first, then plain
    h5ad_path = os.path.join(out_dir, 'stamp_mousebrain_v7a.h5ad')
    if not os.path.exists(h5ad_path):
        h5ad_path = os.path.join(out_dir, 'stamp_mousebrain.h5ad')
    
    if os.path.exists(h5ad_path):
        adata_out = sc.read_h5ad(h5ad_path)
        best_labels = adata_out.obs['stamp_domain'].astype(int).values
        z_stamp_np = adata_out.obsm['X_stamp']
        print(f"Loaded saved mouse brain results from {h5ad_path}")
    else:
        print("No saved mouse brain results found. Need to run training first.")
        return False
    
    labels_true = np.array(adata_out.obs['layer'].astype('category').cat.codes)
    from sklearn.metrics import adjusted_rand_score
    best_ari = adjusted_rand_score(labels_true, best_labels)
    
    # Map clusters to layer names for color consistency
    layer_cats = sorted(set(layer_names))
    layer_name_to_code = {name: i for i, name in enumerate(layer_cats)}
    labels_true_names = np.array([layer_cats[c] for c in labels_true])
    
    mapped_names = map_clusters_to_cell_types(best_labels, labels_true, layer_cats)
    
    unique_domains = sorted(np.unique(best_labels))
    domain_name_map = {d: f'Predicted Domain {d+1}' for d in unique_domains}
    pretty_labels = np.array([domain_name_map[d] for d in best_labels])
    
    # Build color mapping from layer names
    n_layers = len(layer_cats)
    if n_layers <= 10:
        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif n_layers <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        extra = plt.cm.Set3(np.linspace(0, 1, 12))
        base_colors = np.vstack([base_colors, extra])
    
    layer_color_dict = {cat: base_colors[i] for i, cat in enumerate(layer_cats)}
    
    # Build mapping from pretty label -> layer name for color
    pretty_to_layer = {}
    for d in unique_domains:
        mask = best_labels == d
        pretty_to_layer[domain_name_map[d]] = mapped_names[mask][0]
    
    # Plot spatial - use larger spot_size for coordinate scale [0,99]
    coords = Pos
    # For coordinates in [0,99] range, spot size should be much larger
    # A reasonable spot size in data units: ~3-5
    spot_size = 4.0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: predicted
    for i, d in enumerate(unique_domains):
        mask = best_labels == d
        pretty_name = domain_name_map[d]
        layer_name = pretty_to_layer[pretty_name]
        color = layer_color_dict[layer_name]
        axes[0].scatter(coords[mask, 0], coords[mask, 1], 
                       c=[color], s=spot_size**2 * 3, label=pretty_name, edgecolors='none', alpha=0.8)
    
    axes[0].set_title(f'STAMP on P22 Mouse Brain (ARI={best_ari:.3f})', fontsize=12)
    axes[0].set_xlabel('spatial1', fontsize=10)
    axes[0].set_ylabel('spatial2', fontsize=10)
    axes[0].set_aspect('equal')
    axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=8, frameon=False)
    
    # Right: ground truth
    gt_vals = np.array(layer_names)
    for i, cat in enumerate(layer_cats):
        mask = gt_vals == cat
        color = layer_color_dict[cat]
        axes[1].scatter(coords[mask, 0], coords[mask, 1], 
                       c=[color], s=spot_size**2 * 3, label=cat, edgecolors='none', alpha=0.8)
    
    axes[1].set_title('Cortical Layers (Ground Truth)', fontsize=12)
    axes[1].set_xlabel('spatial1', fontsize=10)
    axes[1].set_ylabel('spatial2', fontsize=10)
    axes[1].set_aspect('equal')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=8, frameon=False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'stamp_mousebrain_p22.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved mouse brain spatial plot to {fig_dir}")
    
    # UMAP plot
    adata_plot = adata_out.copy()
    sc.pp.neighbors(adata_plot, n_neighbors=15, n_pcs=z_stamp_np.shape[1], use_rep='X_stamp')
    sc.tl.umap(adata_plot)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    umap_coords = adata_plot.obsm['X_umap']
    
    for i, d in enumerate(unique_domains):
        mask = best_labels == d
        pretty_name = domain_name_map[d]
        layer_name = pretty_to_layer[pretty_name]
        color = layer_color_dict[layer_name]
        axes[0].scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                       c=[color], s=15, label=pretty_name, edgecolors='none', alpha=0.6)
    
    axes[0].set_title(f'STAMP UMAP (ARI={best_ari:.3f})', fontsize=12)
    axes[0].set_xlabel('UMAP1', fontsize=10)
    axes[0].set_ylabel('UMAP2', fontsize=10)
    axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=7, frameon=False)
    
    for i, cat in enumerate(layer_cats):
        mask = gt_vals == cat
        color = layer_color_dict[cat]
        axes[1].scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                       c=[color], s=15, label=cat, edgecolors='none', alpha=0.6)
    
    axes[1].set_title('Ground Truth UMAP', fontsize=12)
    axes[1].set_xlabel('UMAP1', fontsize=10)
    axes[1].set_ylabel('UMAP2', fontsize=10)
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                   fontsize=7, frameon=False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'stamp_umap_mousebrain_p22.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved mouse brain UMAP plot to {fig_dir}")
    
    return True


if __name__ == '__main__':
    fig_dir = '/data/lvyongji/Assignment5/code/figures_v7a'
    
    # Fix simulated datasets D1-D5
    for d in range(1, 6):
        fix_simulated_plot(d, fig_dir)
    
    # Fix mouse brain
    success = fix_mousebrain_plot(fig_dir)
    if not success:
        print("\n[WARNING] Mouse brain results not found. Run run_mousebrain.py first.")
    
    print("\n[DONE] All plots regenerated with fixes.")
