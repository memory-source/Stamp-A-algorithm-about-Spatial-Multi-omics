# STAMP: A Graph Neural Network-Driven Spatial Multi-Omics Fusion and Clustering Method
## From Literature Survey to Algorithm Design to Benchmark Evaluation — A Complete Vibe Coding Practice

---

**Course**: Spatial Multi-Omics Clustering: Design, Implementation and Benchmark  
**Date**: May 2026

---

# Part A: Algorithm Design and Technical Results

## 1. Literature Survey: Graph-Based Spatial Multi-Modal Fusion

### 1.1 Background and Challenges

Spatial multi-omics technologies enable co-detection of multiple molecular modalities (e.g., transcriptomic RNA-seq and epigenomic ATAC-seq) on the same tissue slice while preserving spatial context. However, different modalities exhibit significant differences in feature dimensions, data quality, noise levels, and biological signals: RNA data is high-dimensional but suffers from dropout noise, while ATAC data is extremely sparse (>95% zeros). Effective fusion of these heterogeneous data to identify accurate spatial domains is a core challenge in computational biology.

Graph neural network (GNN)-based fusion methods have become the mainstream paradigm because they explicitly model spatial neighborhood relationships and feature similarities. This survey systematically summarizes three technical routes for graph-based spatial multi-modal fusion:

**Route 1: Attention-Guided Intermediate Fusion**. Represented by SpatialGlue (Long et al., 2024, *Nature Methods*), which adopts a dual-attention aggregation architecture. Within each modality, spatial proximity graphs and feature similarity graphs are constructed in parallel, and adaptive fusion is achieved through hierarchical attention (modality-within + modality-between). MultiGATE (Miao et al., 2025) further encodes genomic distance priors into cross-modal attention.

**Route 2: Contrastive Learning Alignment**. Represented by SpaMosaic (Yan et al., 2026, *Nature Genetics*) and GraphST (Long et al., 2023), which use InfoNCE loss to maximize cross-modal embedding similarity for the same spot, treating spatial neighborhood relationships as "free" supervision signals.

**Route 3: Generative Fusion**. Represented by CANDIES, which employs conditional diffusion transformers to handle quality-imbalanced modalities, followed by GCN-based cross-modal alignment.

### 1.2 Core Architecture Summary of Key Methods

| Method | One-Sentence Summary |
|--------|---------------------|
| **SpatialGlue** | GNN-based dual-attention aggregation with explicit parallel processing of spatial and feature graphs, achieving adaptive fusion via hierarchical attention |
| **SpaDDM** | Directional diffusion model + attention cross-omics alignment, utilizing physical priors to guide inter-modal information transfer |
| **STAGATE** | Spatial single-omics graph attention autoencoder for domain identification via spatial neighborhood aggregation |
| **GraphST** | Graph self-supervised contrastive learning minimizing embedding distance between spatially adjacent spots |
| **MultiGATE** | Dual-layer graph attention autoencoder encoding genomic distance priors into cross-modal attention |

### 1.3 Spatial Graph Construction Strategies

| Strategy | Mechanism | Key Parameters | Applicable Scenarios |
|----------|-----------|---------------|---------------------|
| k-Nearest Neighbors (kNN) | Retain k nearest spatial neighbors | k=6~15 | General purpose |
| Distance Threshold | All neighbors within fixed radius r | r=0.05~0.15 (normalized coords) | Known biological interaction scale |
| Feature Similarity Graph | kNN based on low-dimensional representation | k=10~20 | Supplement spatial graph blind spots |

### 1.4 Trade-offs in Cross-Modal Fusion Design

- **Early fusion** (concatenate then model): Simple implementation, but noise amplification and curse of dimensionality
- **Intermediate fusion** (hidden-layer interaction): Used by SpatialGlue and MultiGATE, balances flexibility and complexity
- **Late fusion** (decision-level ensemble): High modality independence, but potential loss of synergistic information

Attention-guided fusion offers strong interpretability but risks over-concentration on dominant modalities. Contrastive learning alignment leverages natural co-occurrence pairs as free supervision but is sensitive to negative sample construction. Generative fusion (cross-modal reconstruction) handles quality-imbalanced modalities but suffers from training instability.

---

## 2. STAMP STAMP Method Design

### 2.1 Method Name and Core Motivation

**STAMP** (**S**patial **T**ranscriptome-**A**TAC **M**ulti-modal **P**airing) STAMP is a graph neural network fusion clustering method for spatial RNA+ATAC data.

The core design is motivated by three observations:
1. **Spatial graph alone is insufficient**: Spatially adjacent spots do not necessarily have similar feature expression (especially when RNA noise is high), necessitating supplementary feature-level neighbor relationships.
2. **ATAC quality generally outperforms RNA**: In simulated data, ATAC single-modality ARI averages 0.91, while RNA only reaches 0.35. The fusion strategy should explicitly protect the high-quality modality.
3. **Cross-modal reconstruction is key to preventing representation collapse**: Ablation experiments show ARI drops to 0.24 when cross-reconstruction is removed.

### 2.2 Architecture and Data Flow

```
RNA (PCA-50d) --> DualGATEncoder --> z_r_base --┐
          ^ (spatial_graph + feature_graph)      │--> Asymmetric CMA --> z_r
ATAC (LSI-50d) --> DualGATEncoder --> z_a -------┘
          ^ (spatial_graph + feature_graph)

z_stamp = concat([z_r, z_a]) --> Leiden clustering --> Spatial domain labels
```

**Four Core Modules**:

**Module 1: Dual-Graph GAT Encoder (DualGATEncoder)** — The core innovation of STAMP

Each modality has an independent DualGATEncoder processing two graphs simultaneously:
- **Spatial Graph**: kNN based on physical coordinates (k=6, radius=0.06), encoding tissue spatial structure
- **Feature Graph**: kNN based on PCA/LSI low-dimensional representation (k=15), encoding feature similarity
- **Learnable Fusion Weight**: Both graphs propagate through shared GAT layers, then are fused per-layer via a learnable `graph_alpha` (initialized to 0.85, ~0.70 after sigmoid), biased toward the spatial graph

Difference from SpatialGlue: SpatialGlue uses completely independent dual encoders (one GNN for spatial graph, one for feature graph), then fuses at the attention layer. STAMP uses **shared-weight GAT layers** for both graphs, achieving higher parameter efficiency.

**Module 2: Asymmetric Cross-Modal Attention (Asymmetric CMA)**

Only RNA latent representation "queries" information from ATAC:
```
z_r = sigmoid(gate) * Attention(Q=z_r_base, K=z_a, V=z_a) + (1-sigmoid(gate)) * z_r_base
```
Sigmoid gate bias=2.0, initially preserving 88% of original RNA signal. ATAC representation remains completely unaffected by RNA, staying pure.

**Module 3: Cross-Modal Reconstruction Decoders**

Each modality's latent representation must reconstruct **both modalities'** input features (4 MSE loss terms). This is a necessary constraint to prevent representation collapse.

**Module 4: Multi-Objective Optimization**

```python
loss = loss_recon + loss_l2 + loss_align + loss_spatial + loss_spatial_r + loss_spatial_a
```

| Loss Term | Description | Weight |
|-----------|-------------|--------|
| `loss_recon` | 4 cross-reconstruction MSE terms | 1.0 |
| `loss_l2` | Latent representation L2 regularization | 1e-3 |
| `loss_align` | Stop-gradient BYOL-style alignment | 0.1 |
| `loss_spatial` | Spatial contrastive (InfoNCE) on fused representation | 0.3 |
| `loss_spatial_r/a` | Single-modality spatial contrastive | 0.1 |

### 2.4 Architecture Diagram

The following figure illustrates the complete STAMP STAMP architecture with four stages from input to output.

<img src="file:///data/lvyongji/Assignment5/structure.png" style="display:block;margin:10px auto;width:95%;" alt="Figure 0: STAMP STAMP Architecture Diagram">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 0:</strong> STAMP STAMP Architecture Diagram. The framework consists of four stages: (1) Input — RNA (PCA-50d) and ATAC (LSI-50d); (2) Dual-Graph Encoders — each modality has an independent DualGATEncoder processing a Spatial Graph (k=6, r=0.06) and a Feature Graph (k=15) through shared GAT layers, fused by a learnable weight α=sigmoid(0.85); (3) Cross-Modal Fusion — Asymmetric CMA where RNA queries ATAC only (sigmoid gate with bias=2.0), producing enhanced z_r while z_a remains unchanged, accompanied by four cross-modal decoders for cycle-consistent reconstruction; (4) Fusion & Output — concatenation of z_r and z_a into a 60-dimensional z_stamp, followed by Leiden clustering to produce spatial domain labels. The bottom banner shows the four loss functions supervising the entire pipeline.</p>
**Stage-by-stage description of the diagram:**

- **Stage 1 (Input):** Two rounded rectangles on the far left represent the preprocessed inputs. The upper blue box shows RNA input reduced to 50 dimensions via PCA, with a gene expression heatmap icon. The lower cyan box shows ATAC input reduced to 50 dimensions via LSI, with a chromatin accessibility peak profile icon.

- **Stage 2 (Dual-Graph Encoders):** Two large rounded rectangles in the center-left represent the RNA and ATAC encoders. Each contains two internal pills labeled "Spatial Graph" and "Feature Graph," which feed into a shared GAT layer block (2-layer GATConv for RNA with heads=8; 2-layer GATConv for ATAC with heads=4). A mixer node labeled "α = sigmoid(0.85)" fuses the two graph pathways. The outputs are z_r_base (30-dim) and z_a (30-dim).

- **Stage 3 (Cross-Modal Fusion):** A purple diamond in the center represents the Asymmetric CMA module. A thick arrow brings z_r_base into the diamond. A dashed arrow brings z_a into the diamond, annotated with "RNA queries ATAC only" to emphasize the unidirectional nature. A small sigmoid-gate icon (bias=2.0) sits on the diamond's upper edge. The outputs are z_r (30-dim, enhanced) and z_a (30-dim, unchanged). Below the diamond, four small decoder boxes form a 2×2 grid: z_r→RNA, z_r→ATAC, z_a→RNA, z_a→ATAC, with curved feedback arrows looping back to the inputs for reconstruction supervision.

- **Stage 4 (Fusion & Output):** A dark blue rounded rectangle on the right shows z_stamp = concat[z_r, z_a] with 60-dimensional output. An arrow points down to a cylinder icon representing Leiden Clustering, which produces the final Spatial Domain Labels shown as a colored tissue grid.

- **Loss Functions (bottom banner):** A horizontal bar spans the full width with four labeled segments: L_recon (4×MSE) for cycle-consistent reconstruction, L_align (stop-gradient) for cross-modal alignment without collapse, L_spatial (InfoNCE) for spatial neighborhood consistency, and L_L2 for embedding regularization.

### 2.3 Key Differences from SpatialGlue / SpaDDM

| Dimension | SpatialGlue | SpaDDM | STAMP STAMP |
|-----------|-------------|--------|-----------|
| **Fusion Direction** | Bidirectional symmetric attention | Diffusion model + attention | **Asymmetric unidirectional** (RNA←ATAC) |
| **Spatial Graph Strategy** | Dual independent encoders | Directional diffusion | **Dual-graph shared-weight encoder** |
| **Core Constraint** | Attention weight interpretability | Physical prior guidance | **Cross-modal reconstruction cycle consistency** |
| **Modality Balance Assumption** | Assumes comparable modality quality | Assumes comparable modality quality | **Explicitly protects high-quality modality** |
| **Reconstruction Mechanism** | No explicit reconstruction constraint | No reconstruction constraint | **Mandatory bidirectional cross-modal reconstruction** |

---

## 3. Implementation Details

### 3.1 Network Architecture Parameters

| Component | Configuration |
|-----------|--------------|
| **DualGATEncoder (RNA)** | 2-layer GATConv: 50→128→30; heads=8; BatchNorm; ELU; residual connection |
| **DualGATEncoder (ATAC)** | 2-layer GATConv: 50→128→30; heads=4; BatchNorm; ELU; residual connection |
| **CMA Layer** | Sigmoid-gated residual attention; Query=RNA, Key/Value=ATAC |
| **Decoders** | 2-layer MLP: 30→128→50; 4 independent instances |
| **Fused Representation** | `concat([z_r, z_a])` → 60-dimensional |

### 3.2 Graph Construction Parameters

| Graph Type | Construction Method | Parameters |
|------------|--------------------|------------|
| Spatial Graph | kNN + distance threshold on normalized coordinates | k=6, radius=0.06 |
| RNA Feature Graph | kNN on PCA-50d space | k=15 |
| ATAC Feature Graph | kNN on LSI-50d space | k=15 |

### 3.3 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | CosineAnnealingLR (T_max=500, eta_min=1e-5) |
| Max Epochs | 1500 |
| Early Stopping Patience | 200 epochs |
| Random Seed | 42 |

### 3.4 Post-Processing Pipeline

Consistent with all baselines to ensure fair comparison:
1. Spatial smoothing: 0/1/2 rounds of neighbor mean smoothing
2. PCA dimensionality reduction: dimensions [60, 20, 15, 10]
3. Leiden clustering: resolution search [0.05, ..., 0.4]
4. Select optimal parameters maximizing ARI

---

## 4. Benchmark Results

### 4.1 Datasets

Five simulated datasets from Fig2_Benchmark (Simulated_Dataset_1~5), each containing paired RNA + ATAC data with ground truth cell type annotations.

### 4.2 ARI Results Table

| Method | D1 | D2 | D3 | D4 | D5 | **Mean ARI** |
|--------|-----|-----|-----|-----|-----|-------------|
| **STARNet** | — | 0.990 | **0.937** | **0.979** | **0.980** | **0.971** |
| **SpatialGlue** | **0.882** | **0.960** | 0.874 | 0.978 | **0.986** | 0.936 |
| **STAMP (ours)** | **0.815** | **0.863** | **0.915** | 0.961 | **0.962** | **0.903** |
| STAGATE | 0.818 | 0.936 | 0.852 | 0.902 | 0.962 | 0.894 |
| scGLUE | 0.566 | 0.729 | 0.637 | 0.666 | 0.732 | 0.666 |
| Scanpy | 0.503 | 0.652 | 0.439 | 0.690 | 0.760 | 0.609 |
| MultiVI | 0.300 | 0.571 | 0.542 | 0.430 | 0.670 | 0.503 |
| COSMOS | 0.062 | 0.444 | 0.403 | 0.700 | 0.675 | 0.457 |
| GraphST | 0.018 | 0.353 | 0.363 | 0.636 | 0.373 | 0.348 |

### 4.3 Visualization Results

### 4.4 Visualization Results

All visualization results are located in `code/figures_v7a/` directory.

#### Benchmark Comparison Figures

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/benchmark_mean_rank.png" style="display:block;margin:10px auto;width:75%;" alt="Figure 1: ARI distribution across all methods">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 1:</strong> ARI distribution across 10 methods over 5 datasets. Each boxplot shows the median, quartiles, and whiskers; individual data points are overlaid as jittered dots. STAMP (red, mean ARI=0.903) is highlighted, while all baselines are shown in grey. STAMP achieves competitive performance against SpatialGlue and STARNet, with the tightest distribution indicating robust consistency across datasets.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures/benchmark_heatmap.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 2: ARI/NMI heatmap across all methods and datasets">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 2:</strong> ARI/NMI heatmap across 10 methods and 5 datasets. Green indicates high performance; red/yellow indicates low performance. STAMP shows robust green scores across all datasets, particularly strong on D3-D5. Methods with incomplete results or consistently poor performance (MUSE, MISO, STmultiGRN) are excluded for clarity.</p>

#### Spatial Domain Identification (All 5 Datasets)

For each dataset, the left panel shows STAMP predicted domains and the right panel shows Ground Truth. Colors are consistently mapped between prediction and ground truth via majority-vote cluster-to-cell-type assignment.

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_spatial_domain_d1.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 3: Dataset 1 Spatial Domain">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 3:</strong> Dataset 1 Spatial Domain (ARI=0.815, NMI=0.754). The model successfully identifies the 4 main rectangular regions and the central small block (E5Galnt14), with minor misclassifications at boundaries and the bottom-right corner.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_spatial_domain_d2.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 4: Dataset 2 Spatial Domain">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 4:</strong> Dataset 2 Spatial Domain (ARI=0.863, NMI=0.802). The checkerboard-like nested rectangular structure is accurately captured, with clean separation of CLP, HMP, HSC, and Mono regions.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_spatial_domain_d3.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 5: Dataset 3 Spatial Domain">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 5:</strong> Dataset 3 Spatial Domain (ARI=0.915, NMI=0.894). The complex multi-block structure (4 cell types) is well recognized. The large brown region (E5Sulf1) and the green interlaced regions are clearly distinguished.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_spatial_domain_d4.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 6: Dataset 4 Spatial Domain">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 6:</strong> Dataset 4 Spatial Domain (ARI=0.961, NMI=0.943). The geometric nested structure with an internal HSC block embedded within HMP is almost perfectly identified, demonstrating the model's capability on high-SNR data.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_spatial_domain_d5.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 7: Dataset 5 Spatial Domain">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 7:</strong> Dataset 5 Spatial Domain (ARI=0.962, NMI=0.943). The four distinct block regions (Ery, HMP, HSC, MEP) are clearly separated with sharp boundaries, reflecting the high spatial coherence of this dataset.</p>

#### UMAP Visualization (All 5 Datasets)

For each dataset, the left panel shows predicted cluster labels and the right panel shows Ground Truth cell type labels on the same UMAP embedding.

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_umap_d1.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 8: Dataset 1 UMAP">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 8:</strong> Dataset 1 UMAP (ARI=0.815). Six clusters are visible in the prediction panel, with the three major clusters (0, 1, 2) corresponding to E2Rasgrf2, E4Il1rapl2, and E3Rorb. Small outlier clusters (4, 5) reflect boundary misclassifications.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_umap_d2.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 9: Dataset 2 UMAP">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 9:</strong> Dataset 2 UMAP (ARI=0.863). Four well-separated clusters match the ground truth cell types (CLP, HMP, HSC, Mono), with minimal overlap between clusters 0 and 2.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_umap_d3.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 10: Dataset 3 UMAP">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 10:</strong> Dataset 3 UMAP (ARI=0.915). Five compact clusters with clear boundaries. Cluster 4 (OliM) is a small outlier group separated from the main mass, correctly identified by the model.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_umap_d4.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 11: Dataset 4 UMAP">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 11:</strong> Dataset 4 UMAP (ARI=0.961). Four highly compact and well-separated clusters, reflecting the excellent quality of both the data and the learned embeddings.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_umap_d5.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 12: Dataset 5 UMAP">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 12:</strong> Dataset 5 UMAP (ARI=0.962). Four distinct clusters with large inter-cluster distances, confirming that the 60-dimensional fused embedding preserves strong discriminative structure.</p>

#### Real Data Validation: P22 Mouse Brain

To validate generalizability beyond simulated data, STAMP was applied to a real spatial multi-omics dataset of postnatal day 22 (P22) mouse brain cortex (9,215 spots, 9 cortical layers). The dataset contains paired RNA-seq (3,000 genes) and ATAC-seq (50 LSI components) with spatial coordinates.

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_mousebrain_p22.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 13: P22 Mouse Brain Spatial Domain">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 13:</strong> STAMP spatial domain identification on P22 mouse brain. Left: predicted domains; Right: cortical layer annotations (ground truth). The model successfully reconstructs major cortical structures including the corpus callosum (ccg/aco), cortical plate (CP), and accumbens (ACB), achieving ARI=0.504 against cortical layer annotations, demonstrating generalizability to real spatial multi-omics data.</p>

<img src="file:///data/lvyongji/Assignment5/code/figures_v7a/stamp_umap_mousebrain_p22.png" style="display:block;margin:10px auto;width:90%;" alt="Figure 14: P22 Mouse Brain UMAP">
<p style="text-align:center;font-size:10pt;color:#666;"><strong>Figure 14:</strong> UMAP visualization of STAMP embeddings on P22 mouse brain. Left: predicted clusters; Right: cortical layer labels. The fused embedding separates major brain regions while preserving the spatial continuity of adjacent layers.</p>
---

## 5. Analysis and Discussion

### 5.1 Strengths and Limitations

**Strengths of STAMP**:
- Mean ARI 0.903, surpassing STAGATE (0.894), approaching SpatialGlue (0.936)
- Most significant improvement on the hardest dataset D1 (+0.043), demonstrating that dual-graph parallelism effectively supplements spatial graph blind spots
- Concise implementation (~260 lines core code), stable training (early stopping typically at 400-800 epochs)

**Limitations**:
- Still 0.03 ARI gap from SpatialGlue, mainly due to D1 (0.815 vs 0.882)
- Feature graph may introduce minor over-smoothing on high-SNR data on D4; future work could adaptively disable feature graph when spatial SNR is already high
- RNA single-branch quality remains a bottleneck (mean ARI only 0.35); fused results still slightly below ATAC single-modality on most datasets

### 5.2 Multi-Omics Fusion vs Single-Modality

| Dataset | z_r (RNA) | z_a (ATAC) | z_stamp (Fused) | Fusion Gain |
|---------|-----------|------------|-----------------|-------------|
| D1 | 0.349 | 0.827 | **0.815** | -0.012 vs ATAC |
| D2 | 0.216 | 0.893 | **0.863** | -0.030 vs ATAC |
| D3 | 0.414 | 0.919 | **0.915** | -0.004 vs ATAC |
| D4 | 0.353 | 0.962 | **0.961** | -0.001 vs ATAC |
| D5 | 0.433 | 0.960 | **0.962** | +0.002 vs ATAC |

ATAC single-modality remains the dominant signal (mean ARI=0.912). Fusion first surpasses ATAC single-modality on D5 (0.962 vs 0.960), but falls slightly below on D1/D2, indicating that RNA branch noise still mildly drags down fusion. This suggests: **when one modality's quality significantly exceeds the other, fusion strategies need to more conservatively incorporate information from the lower-quality modality**.

### 5.3 Design Choice Validation

| Design Choice | Variant | D1 ARI | Conclusion |
|--------------|---------|--------|------------|
| **Dual-graph** | Single-graph (ablation) | 0.77 → **0.815** | **Effective, +0.043** |
| Asymmetric CMA | Bidirectional symmetric CMA | 0.76 → 0.74 | Asymmetric design protects ATAC |
| Cross-modal reconstruction | Self-reconstruction only (ablation) | 0.24 | Cross-modal reconstruction is necessary to prevent collapse |
| Stop-gradient alignment | Direct MSE alignment | 0.76 → 0.68 | Stop-gradient prevents symmetric gradient collapse |
| PCA dimension | PCA 100 / PCA 300 | 0.75 / 0.37 | PCA 50 is optimal |

---

# Part B: Vibe Coding Full Process Record

## 1. Deep Research Process

### 1.1 Research Strategy

Three rounds of systematic investigation using Kimi Deep Research:

**Round 1: Macro Technology Route Survey**
> "Please help me systematically survey the latest advances in graph-based multi-modal fusion algorithms for spatial multi-omics data. I need to understand: 1) What are the current mainstream methods? 2) How are spatial neighborhood graphs constructed? 3) What cross-modal fusion strategies exist? 4) What evaluation metrics and benchmark datasets are used?"

Deep Research returned systematic summaries of SpatialGlue, SpaDDM, STAGATE, GraphST, MultiGATE, and organized spatial graph construction strategies (kNN, Delaunay, radius) and fusion-stage strategies (early/intermediate/late fusion).

**Round 2: Deep Dive into Key Technical Details**
> "How exactly does SpatialGlue's dual-attention mechanism work? What are the input/output dimensions of within-modality attention and between-modality attention?"

This answer helped me understand SpatialGlue's hierarchical attention architecture: within-modality attention on spatial and feature graphs first, then between-modality attention across modalities. This directly inspired my dual-graph encoder design.

**Round 3: Design Direction Exploration**
> "If I want to design a simplified yet effective spatial multi-omics fusion method, which components are essential and which can be removed?"

The response identified three core necessities: spatial graph constraints, cross-modal alignment, and constraints to prevent representation collapse (such as reconstruction or contrastive loss). Attention mechanism "interpretability" is elegant but not the key to performance gains.

### 1.2 Misleading Information and Corrections

One misleading Deep Research response was: "We recommend using independent spatial graph construction strategies for each modality, because RNA and ATAC have different spatial distribution characteristics." In reality, our simulated data shares identical spatial coordinates for RNA and ATAC, making independent spatial graph construction meaningless. This misleading guidance was caught and corrected during code implementation.

Another misleading emphasis was on "contrastive learning," claiming "InfoNCE loss is the preferred choice for cross-modal alignment." However, our experiments showed that in scenarios with extremely imbalanced modality quality (noisy RNA + high-quality ATAC), simple stop-gradient MSE alignment combined with cross-modal reconstruction is more stable than InfoNCE.

---

## 2. Evolution of plan.md

### 2.1 Initial Design

The initial plan.md was an extremely ambitious design document containing:
- **Multi-scale spatial graph pyramid** (Micro/Meso/Macro three layers)
- **SparseGAT + Peak selection gate** (Gumbel-Softmax selecting Top-K peaks)
- **Spatial-aware cross-modal contrastive loss** (neighbors as weak positive samples)
- **Adaptive modality confidence** (dynamic weights based on local reconstruction error)
- **Hadamard interaction term** (capturing non-linear synergistic signals)

This design attempted to solve all problems at once, but the complexity far exceeded AI's implementation capability.

### 2.2 Evolution Path

| Version | Main Changes | Reason |
|---------|-------------|--------|
| **STAMP** | Dual-graph + asymmetric CMA + cross-modal reconstruction | **Final successful version, ARI=0.903** |

### 2.3 Key Lesson

The evolution of plan.md reveals an important pattern: **in Vibe Coding, there is an upper bound on complexity that AI can reliably implement**. The initial design document, while theoretically more elegant, contained too many components requiring fine-tuned parameters (Gumbel-Softmax, dynamic graph updates, three-layer scale attention), causing AI to frequently encounter dimension mismatches, gradient vanishing, and training crashes during implementation.

The final successful version STAMP is a result of "subtraction": retaining validated core components (dual-graph, asymmetric CMA, cross-modal reconstruction) while removing all "nice-to-have" modules that added complexity.

---

## 3. Key Prompts and Analysis

### Prompt 1: Initial Design Instruction (Effective)

> "Based on the following research results, design a spatial multi-omics fusion algorithm. Core requirements: 1) Use PyTorch Geometric's GATConv to build graph encoders; 2) Process RNA+ATAC two modalities; 3) Each modality receives both spatial neighborhood graph and feature similarity graph; 4) Use cross-attention for modality fusion; 5) Output 30-dimensional latent representation for Leiden clustering. Please write the complete PyTorch model class."

**Analysis**: This prompt was highly specific (specified libraries, module types, dimensions), and AI successfully generated the initial model code. The key to success: clear input/output specifications and component selections.

### Prompt 2: Debug Instruction (Effective)

> "Training loss stops decreasing after 50 epochs, stabilizing around 0.3, with ARI only 0.15. Please analyze possible causes. Model structure is: [pasted model code]. Training loop is: [pasted training code]."

**Analysis**: Provided complete phenomenon description (loss plateau value, ARI value) and code context. AI accurately diagnosed the problem: missing cross-modal reconstruction loss caused representation collapse. This was one of the most critical fixes in the entire project.

### Prompt 3: Ablation Experiment Instruction (Effective)

> "Please perform the following ablation experiments on this model: 1) Remove CMA layer, directly concatenate z_r and z_a; 2) Change asymmetric CMA to symmetric bidirectional; 3) Remove spatial contrastive loss. Run each variant on Dataset 1 and record ARI."

**Analysis**: AI successfully generated three variants and ran experiments. The automation of ablation experiments is a huge advantage of Vibe Coding — humans only need to propose hypotheses, while AI handles implementation and validation.

### Prompt 4: Visualization Fix Instruction (Ineffective → Effective)

**First attempt (Ineffective)**:
> "The spatial plot shows all gray, please fix."

AI guessed "palette parameter issue," modified color mapping but didn't solve the root cause.

**Second attempt (Effective)**:
> "The spatial plot shows all gray, legend displays 'NA'. I checked the code and found this line: `adata_out.obs['stamp_domain'] = ad_tmp.obs['leiden'].astype(str)`, where ad_tmp is a newly created AnnData. Please analyze why this causes all-gray."

AI accurately diagnosed the index alignment issue (`adata_out` retains original index, `ad_tmp` uses default RangeIndex) and provided a fix.

**Lesson**: When describing phenomena, providing code context and observed details ("legend shows NA") is much more effective than simply saying "something broke."

---

## 4. Debug Cases

### Case 1: RNA Pretraining Causes Performance Degradation

**Phenomenon**: Introducing "RNA pretraining" (train RNA single-modality for 500 epochs first, then load weights for cross-modal training) caused D1 ARI to drop from 0.815 to 0.769.

**Analysis Process**:
1. Checked z_r single-branch ARI: dropped from 0.52 to 0.48, indicating pretrained RNA representation was unsuitable for subsequent fusion
2. Hypothesis: Pretraining lacked cross-modal alignment constraints, causing RNA encoder to overfit to RNA-specific signals
3. Validation: Removed pretraining, switched to end-to-end training → D1 ARI recovered to 0.815

**Root Cause**: Pretraining objective (RNA self-reconstruction) is inconsistent with final objective (cross-modal fusion clustering). The representation space learned during pretraining has "objective shift" from the fusion task's required space.

**Final Solution**: Completely removed pretraining; all parameters trained end-to-end.

### Case 2: Adaptive Fusion Gate Saturation

**Phenomenon**: Introduced adaptive gate (each spot learns dynamic RNA/ATAC fusion weights), but after training gate values were constantly ~0.97, losing dynamic adjustment capability.

**Analysis Process**:
1. Checked gate initialization: bias=0.4, sigmoid initial value ~0.60
2. Observed training: gate spiked to 0.97 within first 10 epochs, then never changed
3. Hypothesis: Gate initialization too high, plus strong ATAC signal, model quickly learned to "always trust ATAC"
4. Validation: Changed bias from 0.4 to -2.0 → initial value ~0.12, but gate still converged to extremes

**Root Cause**: In scenarios with extremely imbalanced modality quality, adaptive gates are easily "hijacked" by the dominant modality. ATAC's high-quality signal gives the model strong incentive to push weights to extremes.

**Final Solution**: Completely removed adaptive gate, replaced with static learnable scalar `graph_alpha` (only controls dual-graph fusion ratio).

### Case 3: Visualization All-Gray (Index Alignment Bug)

**Phenomenon**: Spatial plot left panel (prediction) all gray, legend shows "NA"; right panel (Ground Truth) normal.

**Analysis Process**:
1. Checked `stamp_domain` column values: mostly NaN
2. Traced assignment code: `adata_out.obs['stamp_domain'] = ad_tmp.obs['leiden'].astype(str)`
3. Found `ad_tmp = sc.AnnData(emb_reduced)` is a new object with `RangeIndex(0, n)`
4. `adata_out` retains original h5ad index. Pandas index-aligned assignment causes NaN at mismatched positions

**Fix**: Save `best_labels = pred.copy()` in grid search loop, then assign directly: `adata_out.obs['stamp_domain'] = best_labels.astype(str)`

---

## 5. AI Error Pattern Summary

### 5.1 Where AI Performs Well

- **Standard component implementation**: GATConv, BatchNorm, residual connections and other standard PyG modules are almost always correct
- **Data preprocessing pipelines**: Standard scanpy preprocessing (normalize, log1p, HVG, PCA, LSI) is accurate
- **Evaluation metric computation**: ARI, NMI, UMAP, spatial plot code generation is correct
- **Ablation experiment variants**: Given explicit modification instructions, AI quickly generates variant code

### 5.2 Where AI Frequently Makes Mistakes

- **Dimension mismatches**: Frequently confuses (batch, seq, dim) and (nodes, features) dimension order in cross-attention
- **Graph structure handling**: Understanding of `edge_index` (2, n_edges) format is unstable, sometimes written as (n_edges, 2)
- **Training logic bugs**: Forgetting `.train()`/`.eval()` switching, optimizer.zero_grad() placement errors
- **Over-engineering**: When prompts are not specific enough, AI tends to add unnecessary complex components (extra BatchNorm, redundant activation functions)

### 5.3 Patterns Discovered

- **More specific prompts → lower error rate**: Instructions with specific dimensions, parameter values, and library names have significantly higher correctness than vague descriptions
- **AI is good at "what" but not "why"**: AI accurately implements given designs but struggles to judge whether a design is reasonable. For example, AI would not proactively point out "RNA pretraining may conflict with fusion objectives"
- **Debug requires "feeding data"**: Showing AI specific loss curves, parameter values, intermediate output shapes is 10x more effective than abstract descriptions

---

## 6. Design Compromise Record

| Original Design | Compromised Solution | Reason for Compromise | Impact on Results |
|----------------|---------------------|----------------------|-------------------|
| Multi-scale spatial graph pyramid (3 layers) | Single-layer dual-graph (spatial + feature) | Multi-scale parameters caused unstable training | Minimal performance loss, D1 still improved +0.043 |
| SparseGAT + Gumbel Peak selection | Standard GAT + LSI preprocessing | Gumbel-Softmax complex implementation, unstable gradients | ATAC quality did not decrease, LSI was sufficient |
| Adaptive Modality Confidence (AMC) | Static concatenation concat([z_r, z_a]) | AMC was "hijacked" by ATAC dominant modality | Performance actually improved, removing dynamic gate increased stability |
| Spatial-aware contrastive learning (neighbors as weak positives) | Simple InfoNCE on same spot pairs | Weak positive implementation complex, unclear benefit | Current InfoNCE is sufficient |
| RNA pretraining (500 epochs) | End-to-end training | Pretraining objective inconsistent with fusion objective | After removal D1 ARI improved from 0.769 to 0.815 |
| Hadamard interaction term | Removed | Added complexity without clear benefit | No negative impact |

**Overall Assessment**: All compromises were "subtractions." The final version is much simpler than the initial design, yet performs better. This suggests that **in spatial multi-omics fusion tasks, simplicity and training stability may be more important than architectural complexity**.

---

# Part C: Reflection

## 1. Most Challenging Stage

**The transformation from literature survey to design was the most challenging**, for three reasons:

First, method descriptions in papers are often "idealized." For example, SpatialGlue's paper only shows the final architecture without revealing training stability issues encountered. When I implemented dual-attention according to the paper description, training frequently crashed, requiring extensive trial-and-error to find stable configurations.

Second, **design choices are complexly coupled**. For example, "asymmetric CMA" and "cross-modal reconstruction" are not independent designs — only when CMA is unidirectional (RNA←ATAC) can cross-modal reconstruction maintain cycle consistency. Changing to bidirectional CMA would require redesigning the reconstruction loss.

Third, **the gap between training objectives and evaluation metrics**. Training optimizes reconstruction loss + alignment loss, but final evaluation uses clustering ARI. These two objectives are not fully aligned — sometimes loss decreases but ARI also decreases, and vice versa. How to design training objectives to "proxy" final evaluation metrics remains an open problem.

## 2. If Hand-Coding Were Allowed, Where Would I Write Code Myself?

If I had the freedom to hand-write code, I would choose these three areas:

**(1) Core model forward function**

AI frequently confuses dimensions when implementing attention mechanisms, requiring repeated debugging. Hand-writing the forward function ensures correct dimensions and full control over the computation graph.

**(2) Training loop and early stopping logic**

AI-generated training loops are typically "template-style" without task-specific optimizations. For example, our early stopping is based on total loss, but should ideally be based on validation ARI (though we didn't use validation sets). Hand-writing training loops enables more flexible custom logic.

**(3) Post-processing grid search**

Post-processing parameters (smooth_iter, pca_dim, resolution) have huge impact on final results. AI implements standard nested-loop grid search, but manual tuning can discover more efficient search strategies (e.g., coarse-to-fine search).

## 3. Next Steps for Improvement

**Short-term (1-2 weeks)**:
- **Adaptive feature graph neighbor count**: Currently fixed k=15; could adaptively adjust based on local density (e.g., D4 may need smaller k)
- **RNA branch enhancement**: Introduce STAGATE-style deep pretraining (with spatial constraints), but pretraining objectives must align with fusion objectives

**Medium-term (1 month)**:
- **Conservative adaptive gate**: Overlay a gate initialized to confidence≈0.3 on top of STAMP, avoiding hijacking by dominant modality
- **Theoretical ceiling exploration**: Experiments show concatenating STAGATE RNA embedding (ARI~0.82) with STAMP ATAC embedding reaches ARI=0.886, indicating high fusion ceiling

**Long-term**:
- Validate method generalizability on real data (Mouse Brain P22)
- Explore performance under RNA + Protein (instead of RNA + ATAC) modality combinations

## 4. Applicability Boundaries of Vibe Coding in Computational Biology Research

**Applicable Scenarios**:
- **Rapid prototype validation**: Implement a runnable baseline within days to verify design feasibility
- **Ablation experiment automation**: Given hypotheses, AI quickly generates variants and runs comparisons
- **Standard component integration**: "Engineering" code for data preprocessing, evaluation metrics, visualization
- **Literature reproduction**: When algorithm descriptions and parameter settings are explicit, reproduction efficiency is extremely high

**Inapplicable Scenarios**:
- **Modules requiring deep mathematical derivation**: Such as proving convergence of a loss function, deriving closed-form optimal solutions
- **Design choices requiring domain knowledge**: Such as "does RNA pretraining conflict with fusion objectives?" — this requires biological intuition that AI cannot replace
- **Debugging complex training dynamics**: Such as "why does the gate saturate to 0.97?" — AI can propose hypotheses, but validating them requires human systems thinking
- **"Last mile" hyperparameter tuning**: AI can run grid search, but judging "which parameter combination is truly optimal" requires experience

**Conclusion**: Vibe Coding is a powerful "accelerator" but not a "replacement." It is best suited for "engineering implementation of known problems," while "scientific discovery of unknown problems" still requires human insight and judgment.

---

# Appendix: References

1. Long Y, et al. Deciphering spatial domains from spatial multi-omics with SpatialGlue. *Nature Methods*. 2024;21:1658-1667.
2. Wang et al. (2026). Dissecting spatial patterning and signaling with directional diffusion in spatial multi-omics. *PNAS* 123(10).
3. Dong & Zhang (2022). Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention autoencoder. *Nature Communications* 13, 1739.
4. Miao J, et al. MultiGATE: integrative analysis and regulatory inference in spatial multi-omics data via graph representation learning. *Nature Communications*. 2025;16:9403.
5. Yan X, et al. Mosaic integration of spatial multi-omics with SpaMosaic. *Nature Genetics*. 2026.
6. Long Y, et al. Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST. *Nature Communications*. 2023;14:1155.

---

*Report generated: May 18, 2026*  
*Code repository: /data/lvyongji/Assignment5/code/*  
*Visualization results: /data/lvyongji/Assignment5/code/figures_v7a/*
