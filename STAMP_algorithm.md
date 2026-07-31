# STAMP: Spatial Transcriptome-ATAC Multi-scale Paired Network
## 一种面向空间多组学数据的无监督融合与聚类算法

> **适用数据结构**：基于 `/data/lvyongji/Assignment5/Fig2_Benchmark` 中的模拟数据格式设计，直接兼容 `AnnData` 对象。  
> **输入**：配对的 `RNA (n_spots × n_genes)` + `ATAC (n_spots × n_peaks)`，以及空间坐标 `obsm['spatial']`。  
> **输出**：统一低维嵌入 `obsm['X_stamp']`、空间聚类标签 `obs['stamp_domain']`、模态置信度权重 `obsm['stamp_alpha']`、以及推断的 Peak-Gene 关联矩阵。

---

## 1. 算法定位与核心动机

现有方法（如 SpatialGlue、scGLUE、COSMOS）在空间多组学整合中主要面临以下问题：

1. **空间图尺度单一**：大多使用固定 kNN 或固定半径构建空间图，难以同时捕捉局部细胞邻域（k=5）和组织域结构（k=30）。
2. **ATAC 极端稀疏性被忽视**：241K peaks 中 >95% 为 0，标准 GCN 在此极度稀疏图上会出现过度平滑和梯度消失。
3. **跨模态对齐信号单一**：仅对齐同一 spot 的 RNA/ATAC，忽略了空间邻居的跨模态一致性（例如，邻近 spot 的表观状态往往相似）。
4. **模态融合权重静态**：多数方法（如 MUSE）对全部细胞使用固定融合系数，无法处理局部区域 RNA 质量高、另一区域 ATAC 信号强的异质性。

**STAMP 的解决思路**：
- 构建**多尺度空间图金字塔**（Micro/Meso/Macro），分别对应细胞邻域、组织域、全局结构。
- 为 ATAC 设计**稀疏图注意力编码器 (SparseGAT)**，仅在 peak 共现的子图上传播消息。
- 引入**空间-感知跨模态对比损失 (Spatial-aware InfoNCE)**，将正负样本从“同一点对”扩展为“空间邻域内跨模态对齐”。
- 提出**基于局部重构误差的自适应模态置信度 (Adaptive Modality Confidence, AMC)**，每个 spot 独立学习 RNA 和 ATAC 的动态权重。

---

## 2. 符号定义与数据结构假设

基于用户数据结构中 `AnnData` 的标准格式，定义如下：

| 符号 | 维度 | 来源 | 含义 |
|------|------|------|------|
| $\mathcal{V}$ | $n=1296$ | `adata_rna.n_obs` | spot/cell 集合 |
| $\mathbf{X}_R$ | $\mathbb{R}^{n \times d_R}$ | `adata_rna.X` (HVG 后 $d_R=3000$) | RNA 表达矩阵（log1p 后） |
| $\mathbf{X}_A$ | $\{0,1\}^{n \times d_A}$ | `adata_atac.X` ($d_A \approx 241757$) | ATAC Peak 二值/计数矩阵 |
| $\mathbf{S}$ | $\mathbb{R}^{n \times 2}$ | `obsm['spatial']` | 空间坐标（已归一化到 $[0,1]^2$） |
| $y_i$ | $\{1,\dots,C\}$ | `obs['cell_type']` | 第 $i$ 个 spot 的真实细胞类型（评估用） |
| $\mathcal{G}^{(l)}$ | 多尺度图 | 算法构建 | 第 $l$ 层空间图，$l \in \{1,2,3\}$ |

---

## 3. 算法架构：四阶段流水线

### Stage 1: 多尺度空间图金字塔构建

不构建单一空间图，而是基于空间坐标 $\mathbf{S}$ 构建三层图：

1. **Micro-graph $\mathcal{G}^{(1)}$**：`k=6` + `radius=0.05`，捕获最直接的物理邻居（类似 Visium 的 6 邻域）。
2. **Meso-graph $\mathcal{G}^{(2)}$**：`k=15` + `radius=0.12`，捕获局部组织微环境。
3. **Macro-graph $\mathcal{G}^{(3)}$**：`k=50` + `radius=0.25`，捕获大尺度组织域结构。

**边权重定义**（高斯核）：
$$w_{ij}^{(l)} = \exp\left(-\frac{\|\mathbf{s}_i - \mathbf{s}_j\|^2}{2\sigma_l^2}\right) \cdot \mathbb{1}[(i,j) \in \mathcal{E}^{(l)}]$$
其中 $\sigma_l$ 与每层半径成正比。

### Stage 2: 模态内异质编码器

#### 2a. RNA 编码器：Multi-scale GAT

将 $\mathbf{X}_R$ 输入三层 GAT，每层对应一个尺度的空间图：

$$\mathbf{H}_R^{(l)} = \text{GAT}^{(l)}(\mathbf{X}_R, \mathcal{G}^{(l)}) \in \mathbb{R}^{n \times h}$$

然后将三层表示通过 **尺度注意力门控** 融合：

$$\mathbf{Z}_R^{intra} = \sum_{l=1}^{3} \alpha_R^{(l)} \cdot \mathbf{H}_R^{(l)}, \quad \alpha_R^{(l)} = \text{Softmax}(\mathbf{q}_R^{\top} \mathbf{H}_R^{(l)})$$

其中 $\mathbf{q}_R$ 是可学习的查询向量。这允许模型自动决定哪些 spot 需要微尺度、哪些需要宏尺度信息。

#### 2b. ATAC 编码器：SparseGAT + Peak Selection

ATAC 维度 $d_A \approx 241K$ 直接输入 GNN 不可行。STAMP 先执行一个**可学习的稀疏投影**：

1. **Peak 选择门**：学习一个稀疏向量 $\mathbf{m} \in \{0,1\}^{d_A}$（通过 Gumbel-Softmax 或 Top-K 选择），仅保留 $K=10000$ 个高信息量 peaks。
   $$\tilde{\mathbf{X}}_A = \mathbf{X}_A \odot \mathbf{m}$$

2. **Peak-Spot 共现子图**：两个 spot 在 SparseGAT 中有边，仅当它们在 $\tilde{\mathbf{X}}_A$ 中有至少一个共同开放的 peak。这避免了在极度稀疏图上传播无意义消息。

3. **SparseGAT 消息传播**：
   $$\mathbf{Z}_A^{intra} = \text{SparseGAT}(\tilde{\mathbf{X}}_A, \mathcal{G}^{(2)})$$
   这里统一使用 Meso-scale 图，因为 ATAC 信号本身较稀疏，Micro-scale 邻居过少会导致图不连通。

### Stage 3: 跨模态空间-感知对比对齐

这是 STAMP 的核心创新。我们不只对齐同一 spot 的 RNA/ATAC，而是利用**空间平滑先验**扩展对比学习：

#### 3a. 投影到统一潜在空间

将模态内编码结果投影到共享的 $d_z$ 维空间（如 $d_z=30$）：
$$\mathbf{z}_R^i = f_R(\mathbf{Z}_R^{intra}[i]), \quad \mathbf{z}_A^i = f_A(\mathbf{Z}_A^{intra}[i])$$

#### 3b. 空间-感知正样本对定义

对于 spot $i$，定义其 RNA 嵌入的**正样本集合**为其 ATAC 嵌入的：
- **强正样本**：同一 spot $i$ 本身（$j=i$）。
- **弱正样本**：空间邻居 $j \in \mathcal{N}^{(1)}(i)$（Micro-graph 邻居）。

**加权 InfoNCE 损失**：
$$\mathcal{L}_{\text{contrast}} = -\sum_{i=1}^{n} \log \frac{\sum_{j \in \mathcal{P}(i)} \lambda_{ij} \exp(\text{sim}(\mathbf{z}_R^i, \mathbf{z}_A^j)/\tau)}{\sum_{k=1}^{n} \exp(\text{sim}(\mathbf{z}_R^i, \mathbf{z}_A^k)/\tau)}$$

其中：
- $\mathcal{P}(i) = \{i\} \cup \mathcal{N}^{(1)}(i)$ 是正样本集合。
- $\lambda_{ii}=1.0$（强对齐），$\lambda_{ij}=0.5$（弱对齐，邻居）。
- $\text{sim}(\cdot, \cdot)$ 是余弦相似度，$\tau=0.1$ 为温度参数。

**生物学直觉**：表观遗传状态（ATAC）通常在空间上比转录本更慢变。因此允许邻近 spot 的 ATAC 作为正样本，能提供更稳定的跨模态监督信号，缓解 ATAC 的 drop-out 问题。

#### 3c. 模态间互注意力（隐式 Peak-Gene 关联）

不依赖外部 GTF 文件，直接从数据学习 Peak-Gene 关联：

$$\mathbf{M}_{PG} = \text{Softmax}\left(\frac{(\mathbf{X}_R \mathbf{W}_Q)(\tilde{\mathbf{X}}_A \mathbf{W}_K)^{\top}}{\sqrt{d_z}}\right) \in \mathbb{R}^{n \times n}$$

然后利用 $\mathbf{M}_{PG}$ 作为跨模态注意力，生成增强的 RNA 表示：
$$\mathbf{Z}_R^{cross} = \mathbf{M}_{PG} \cdot \mathbf{Z}_A^{intra}$$

最终 RNA 表示为模态内和跨模态的拼接：
$$\mathbf{Z}_R^{final} = [\mathbf{Z}_R^{intra} \| \mathbf{Z}_R^{cross}]$$

对应的 ATAC 也做对称操作，得到 $\mathbf{Z}_A^{final}$。

### Stage 4: 自适应模态置信度融合与聚类

#### 4a. 局部重构误差计算

对每个 spot $i$，分别用一个小型 MLP 解码器重构原始特征：
$$\hat{\mathbf{x}}_R^i = \text{Dec}_R(\mathbf{z}_R^i), \quad \hat{\mathbf{x}}_A^i = \text{Dec}_A(\mathbf{z}_A^i)$$

重构损失：
$$\mathcal{L}_{\text{recon}} = \|\mathbf{X}_R - \hat{\mathbf{X}}_R\|_F^2 + \|\mathbf{X}_A - \hat{\mathbf{X}}_A\|_F^2$$

对每个 spot，记录其局部重构误差 $e_R^i$ 和 $e_A^i$。

#### 4b. 自适应模态权重 (AMC)

基于重构误差计算每个 spot 的模态置信度：
$$\beta_R^i = \frac{\exp(-e_R^i / T)}{\exp(-e_R^i / T) + \exp(-e_A^i / T)}, \quad \beta_A^i = 1 - \beta_R^i$$

其中 $T$ 是温度超参数。重构误差越小，该模态在该 spot 上的权重越高。这能自动处理局部区域的模态质量差异（例如某区域 ATAC dropout 严重时，自动提高 RNA 权重）。

#### 4c. 最终融合嵌入

$$\mathbf{z}_i^{stamp} = \beta_R^i \cdot \mathbf{z}_R^i + \beta_A^i \cdot \mathbf{z}_A^i + \gamma \cdot (\mathbf{z}_R^i \odot \mathbf{z}_A^i)$$

最后一项是 **Hadamard 交互项**，用于捕捉模态间的非线性协同信号（如某些 peak 开放恰好与基因高表达同时出现）。$\gamma$ 是可学习标量。

#### 4d. 聚类

对 $\mathbf{Z}^{stamp}$ 进行 KNN + Leiden 聚类（或 Louvain），得到空间域标签。

---

## 4. 总体损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{contrast}} + \lambda_2 \mathcal{L}_{\text{graph}} + \lambda_3 \mathcal{L}_{\text{sparse}}$$

- $\mathcal{L}_{\text{graph}}$：图平滑正则化（空间近邻在潜在空间中也应该近邻）。
- $\mathcal{L}_{\text{sparse}}$：ATAC peak 选择门的 L0/L1 稀疏惩罚（鼓励只选少量 peaks）。
- 默认超参：$\lambda_1=1.0, \lambda_2=0.1, \lambda_3=0.01$。

---

## 5. 伪代码实现框架

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import scanpy as sc

class STAMP(nn.Module):
    def __init__(self, d_rna, d_atac, hidden=64, latent=30, k_peaks=10000):
        super().__init__()
        # Stage 1: Multi-scale RNA encoder
        self.gat_rna_1 = GATConv(d_rna, hidden, heads=4)
        self.gat_rna_2 = GATConv(d_rna, hidden, heads=4)
        self.gat_rna_3 = GATConv(d_rna, hidden, heads=4)
        self.scale_gate = nn.Linear(hidden*4, 3)  # 3 scales
        
        # Stage 2: ATAC Sparse encoder
        self.peak_selector = nn.Linear(d_atac, k_peaks)  # Top-K/Gumbel selection
        self.sparse_gat = GATConv(k_peaks, hidden, heads=4)
        
        # Stage 3: Cross-modal projectors
        self.proj_rna = nn.Linear(hidden*4, latent)
        self.proj_atac = nn.Linear(hidden*4, latent)
        
        # Stage 4: Cross-attention & decoders
        self.cross_attn_q = nn.Linear(latent, latent)
        self.cross_attn_k = nn.Linear(latent, latent)
        self.dec_rna = nn.Linear(latent, d_rna)
        self.dec_atac = nn.Linear(latent, d_atac)
        
        self.gamma = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x_rna, x_atac, edge_index_1, edge_index_2, edge_index_3):
        # --- Multi-scale RNA encoding ---
        h_r1 = torch.relu(self.gat_rna_1(x_rna, edge_index_1))
        h_r2 = torch.relu(self.gat_rna_2(x_rna, edge_index_2))
        h_r3 = torch.relu(self.gat_rna_3(x_rna, edge_index_3))
        
        gates = torch.softmax(self.scale_gate(h_r1 + h_r2 + h_r3), dim=-1)
        z_r_intra = gates[:,0:1]*h_r1 + gates[:,1:2]*h_r2 + gates[:,2:3]*h_r3
        
        # --- ATAC sparse encoding ---
        peak_scores = self.peak_selector(x_atac)  # (n, k_peaks)
        peak_mask = torch.topk(peak_scores, k=self.k_peaks, dim=-1)[1]
        x_atac_sparse = x_atca * peak_mask.float()  # simplified
        z_a_intra = torch.relu(self.sparse_gat(x_atac_sparse, edge_index_2))
        
        # --- Projection ---
        z_r = self.proj_rna(z_r_intra)
        z_a = self.proj_atac(z_a_intra)
        
        # --- Cross-modal attention ---
        M_pg = torch.softmax(
            (self.cross_attn_q(z_r) @ self.cross_attn_k(z_a).T) / sqrt(latent), dim=-1
        )
        z_r_cross = M_pg @ z_a
        z_r_final = torch.cat([z_r, z_r_cross], dim=-1)
        
        # --- Reconstruction & AMC ---
        recon_r = self.dec_rna(z_r)
        recon_a = self.dec_atac(z_a)
        
        err_r = ((x_rna - recon_r)**2).sum(dim=1, keepdim=True)
        err_a = ((x_atac - recon_a)**2).sum(dim=1, keepdim=True)
        beta_r = torch.exp(-err_r/0.1) / (torch.exp(-err_r/0.1) + torch.exp(-err_a/0.1))
        beta_a = 1 - beta_r
        
        z_stamp = beta_r * z_r + beta_a * z_a + self.gamma * (z_r * z_a)
        return z_stamp, recon_r, recon_a, beta_r, M_pg
```

---

## 6. 与 Fig2_Benchmark 现有方法的对比

| 特性 | STAMP | SpatialGlue | scGLUE | MultiVI | COSMOS | MUSE |
|------|-------|-------------|--------|---------|--------|------|
| **多尺度空间图** | ✅ 三层金字塔 | ❌ 单层 kNN | ❌ 无空间 | ❌ 无空间 | ✅ WNN 有空间 | ❌ 无空间 |
| **ATAC 稀疏性处理** | ✅ SparseGAT+Peak 选择 | ❌ 直接 LSI | ❌ 需 GTF | ✅ VAE 隐式处理 | ❌ 直接高维输入 | ❌ PCA 降维后输入 |
| **空间感知对比学习** | ✅ 邻居作为弱正样本 | ❌ 仅模态内对比 | ❌ 对抗对齐 | ❌ ELBO | ❌ WNN 最近邻 | ✅ 双视图对齐 |
| **动态模态权重** | ✅ 基于局部重构误差 | ✅ 全局 alpha | ❌ 固定 | ❌ 固定 | ❌ 固定 | ❌ 固定 |
| **无外部注释 Peak-Gene** | ✅ 跨注意力学习 | ❌ 无 | ❌ 必须 GTF | ❌ 无 | ❌ 无 | ❌ 无 |
| **计算复杂度** | 中（Peak 选择可控） | 中 | 高（需训 GAN） | 高 | 低 | 低 |

---

## 7. 在该数据上的使用流程

### 7.1 输入准备

基于 `Generate_simulated_data.ipynb` 产生的数据结构：

```python
import scanpy as sc
import torch

# 读取配对数据
adata_rna = sc.read_h5ad('Original_Simulated_Data/Simulated_Dataset_1/SimulatedData_1_rna.h5ad')
adata_atac = sc.read_h5ad('Original_Simulated_Data/Simulated_Dataset_1/SimulatedData_1_atac.h5ad')

# RNA 预处理（与 Scanpy 流程一致）
sc.pp.highly_variable_genes(adata_rna, n_top_genes=3000)
adata_rna = adata_rna[:, adata_rna.var.highly_variable].copy()
sc.pp.scale(adata_rna)

# ATAC 预处理
sc.pp.filter_genes(adata_atac, min_cells=1)
# 注意：ATAC 不做 HVG，而是交给 STAMP 的 peak_selector 自动选择

# 将 ATAC 空间坐标同步到 RNA（它们已经是同一批细胞）
adata_atac.obsm['spatial'] = adata_rna.obsm['spatial']
```

### 7.2 运行 STAMP

```python
from stamp import STAMP, build_multiscale_graphs  # 假设的接口

# 构建三层空间图
edge_index_1, edge_index_2, edge_index_3 = build_multiscale_graphs(
    adata_rna.obsm['spatial'], 
    k_list=[6, 15, 50], 
    radius_list=[0.05, 0.12, 0.25]
)

# 初始化模型
model = STAMP(
    d_rna=adata_rna.n_vars,
    d_atac=adata_atac.n_vars,
    hidden=64,
    latent=30,
    k_peaks=10000
)

# 训练
z_stamp, labels, alpha, M_pg = model.fit(
    adata_rna.X, 
    adata_atac.X,
    edge_index_1, edge_index_2, edge_index_3,
    epochs=500, lr=1e-3
)

# 保存结果
adata_rna.obsm['X_stamp'] = z_stamp
adata_rna.obsm['stamp_alpha'] = alpha  # RNA 模态权重
adata_rna.obs['stamp_domain'] = labels
adata_rna.uns['stamp_peak_gene_attention'] = M_pg  # 推断的关联
adata_rna.write_h5ad('Processed_Simulated_Data/Simulated_Dataset_1/stamp_multiomics.h5ad')
```

### 7.3 结果解读

- **`X_stamp`**: 30 维统一嵌入，可直接用于 UMAP 可视化或输入 Leiden 聚类。
- **`stamp_alpha`**: 形状 `(1296, 1)`，值域 `[0,1]`。越接近 1 表示该 spot 的 RNA 信号更可靠/更具信息量；越接近 0 表示 ATAC 更可靠。
- **`stamp_peak_gene_attention`**: 形状 `(1296, 1296)` 的注意力矩阵（spot-spot 级别），可进一步聚合为 Peak-Gene 关联。

---

## 8. 算法优势总结

1. **数据结构原生适配**：完全基于 `AnnData` 设计，直接兼容 `Fig2_Benchmark` 中的 `obs`, `obsm`, `layers` 标准结构。
2. **空间-感知对比**：突破了传统跨模态对比仅对齐“同一点对”的局限，利用空间平滑先验提升 ATAC 稀疏信号下的对齐稳定性。
3. **自适应融合**：每个 spot 独立学习模态权重，比全局固定权重更能反映真实生物学场景中模态质量的局部异质性。
4. **无需外部注释**：不依赖 GTF 文件即可学习 Peak-Gene 关联，降低了在新型物种或参考基因组不完善时的使用门槛。
5. **多尺度感知**：通过图金字塔同时捕获细胞邻域、组织域和全局结构，避免单一尺度导致的边界模糊或过度分割。
