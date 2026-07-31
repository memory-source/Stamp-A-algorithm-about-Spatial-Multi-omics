"""
STAMP: Spatial Transcriptome-ATAC Multi-scale Paired Network
Utilities for graph construction, preprocessing, and evaluation.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph


def build_knn_graph(spatial, k, radius=None, include_self=False):
    """Build kNN graph from spatial coordinates."""
    adj = kneighbors_graph(spatial, n_neighbors=k, mode='connectivity', include_self=include_self)
    if radius is not None:
        dist_adj = kneighbors_graph(spatial, n_neighbors=k, mode='distance', include_self=include_self)
        dist_adj = dist_adj.tocoo()
        mask = dist_adj.data <= radius
        row, col, data = dist_adj.row[mask], dist_adj.col[mask], np.ones(mask.sum())
        adj = sp.csr_matrix((data, (row, col)), shape=adj.shape)
    edge_index = from_scipy_sparse_matrix(adj)[0]
    return edge_index


def compute_spatial_neighbors(spatial, k):
    """Return list of neighbor indices for each node."""
    tree = cKDTree(spatial)
    _, idx = tree.query(spatial, k=k+1)
    return idx[:, 1:]  # exclude self


def preprocess_atac(adata_atac, n_top_peaks=5000):
    """Select top accessible peaks for ATAC to reduce dimensionality (fast count-based selection)."""
    X = adata_atac.X
    if sp.issparse(X):
        peak_counts = np.asarray(X.sum(axis=0)).ravel()
    else:
        peak_counts = np.sum(X, axis=0)
    peak_counts = np.asarray(peak_counts).ravel()
    selected = np.argsort(peak_counts)[-n_top_peaks:]
    selected = np.sort(selected)
    adata_atac = adata_atac[:, selected].copy()
    return adata_atac


def preprocess_rna(adata_rna, n_top_genes=3000):
    """Standard RNA preprocessing."""
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_top_genes)
    adata_rna = adata_rna[:, adata_rna.var.highly_variable].copy()
    sc.pp.scale(adata_rna)
    return adata_rna


def spatial_smooth_embedding(emb, edge_index):
    """
    Mean-smooth embeddings over spatial graph (pre-clustering refinement).
    emb: (n_nodes, n_dims) numpy array
    edge_index: (2, n_edges) torch.Tensor or numpy array
    """
    import torch
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    n = emb.shape[0]
    smoothed = emb.copy()
    # compute neighbor counts and sums
    neighbor_sum = np.zeros_like(emb)
    neighbor_cnt = np.zeros(n)
    src, dst = edge_index
    for s, d in zip(src, dst):
        neighbor_sum[s] += emb[d]
        neighbor_cnt[s] += 1
    # add self
    neighbor_sum += emb
    neighbor_cnt += 1
    smoothed = neighbor_sum / neighbor_cnt[:, None]
    return smoothed


def lsi_atac(X, n_components=50, random_state=42):
    """
    Latent Semantic Indexing for ATAC: TF-IDF followed by TruncatedSVD.
    X can be dense or sparse (n_cells x n_peaks).
    Returns array of shape (n_cells, n_components).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    if sp.issparse(X):
        X = X.copy()
    else:
        X = sp.csr_matrix(X)
    # TF-IDF
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X)
    # Truncated SVD
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    embedding = svd.fit_transform(X_tfidf)
    return embedding


def evaluate_clustering(labels_pred, labels_true):
    """Compute ARI and NMI."""
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return ari, nmi
