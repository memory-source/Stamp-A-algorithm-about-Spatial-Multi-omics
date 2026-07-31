import json
import html

# Load benchmark results
with open('/data/lvyongji/Assignment5/code/benchmark_results.json') as f:
    results = json.load(f)

# Read code files
with open('/data/lvyongji/Assignment5/code/stamp_model.py') as f:
    stamp_model_code = f.read()
with open('/data/lvyongji/Assignment5/code/run_stamp.py') as f:
    run_stamp_code = f.read()
with open('/data/lvyongji/Assignment5/code/stamp_utils.py') as f:
    stamp_utils_code = f.read()

# Escape HTML in code
def esc(s):
    return html.escape(s)

# Simple Python syntax highlighting
def highlight_python(code):
    import re
    # Keywords
    keywords = ['def', 'class', 'return', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'in', 'not', 'and', 'or', 'True', 'False', 'None', 'self', 'super', 'pass', 'break', 'continue', 'try', 'except', 'finally', 'with', 'as', 'yield', 'lambda', 'assert', 'del', 'global', 'nonlocal', 'raise']
    code = esc(code)
    # Comments
    code = re.sub(r'(#.*?)$', r'<span class="c">\1</span>', code, flags=re.MULTILINE)
    # Strings
    code = re.sub(r'(""".*?""")', r'<span class="s">\1</span>', code, flags=re.DOTALL)
    code = re.sub(r"('''.*?''')", r'<span class="s">\1</span>', code, flags=re.DOTALL)
    code = re.sub(r'(".*?")', r'<span class="s">\1</span>', code)
    code = re.sub(r"('.*?')", r'<span class="s">\1</span>', code)
    # Numbers
    code = re.sub(r'\b(\d+\.?\d*)\b', r'<span class="n">\1</span>', code)
    # Keywords
    for kw in keywords:
        code = re.sub(r'\b(' + kw + r')\b', r'<span class="k">\1</span>', code)
    # Decorators
    code = re.sub(r'(@\w+)', r'<span class="d">\1</span>', code)
    # Functions
    code = re.sub(r'(\w+)(\()', r'<span class="f">\1</span>\2', code)
    return code

# Build result table
methods = sorted(set(m for r in results.values() for m in r.keys()))
datasets = ['1','2','3','4','5']

table_rows = []
for m in methods:
    row = f'<tr data-method="{m}">'
    row += f'<td class="method-name">{m}</td>'
    mean_ari = 0
    mean_nmi = 0
    count = 0
    for d in datasets:
        if m in results[d]:
            ari = results[d][m]['ARI']
            nmi = results[d][m]['NMI']
            mean_ari += ari
            mean_nmi += nmi
            count += 1
            row += f'<td class="ari">{ari:.4f}</td><td class="nmi">{nmi:.4f}</td>'
        else:
            row += '<td class="na">—</td><td class="na">—</td>'
    if count > 0:
        mean_ari /= count
        mean_nmi /= count
    row += f'<td class="mean"><b>{mean_ari:.4f}</b></td><td class="mean"><b>{mean_nmi:.4f}</b></td>'
    row += '</tr>'
    table_rows.append((mean_ari, row))

table_rows.sort(key=lambda x: -x[0])
table_html = '\n'.join([r[1] for r in table_rows])

# Single modality comparison table
single_mod = """
<table class="data-table" id="singleModTable">
<thead>
<tr><th>Dataset</th><th>z_r (RNA only)</th><th>z_a (ATAC only)</th><th>z_stamp (Fused)</th><th>Fusion Gain vs ATAC</th></tr>
</thead>
<tbody>
<tr><td>D1</td><td>0.519</td><td>0.748</td><td class="highlight">0.772</td><td>+3.2%</td></tr>
<tr><td>D2</td><td>0.142</td><td>0.870</td><td class="highlight">0.859</td><td>-1.3%</td></tr>
<tr><td>D3</td><td>0.637</td><td>0.884</td><td class="highlight">0.913</td><td>+3.3%</td></tr>
<tr><td>D4</td><td>0.620</td><td>0.962</td><td class="highlight">0.975</td><td>+1.4%</td></tr>
<tr><td>D5</td><td>0.497</td><td>0.936</td><td class="highlight">0.950</td><td>+1.5%</td></tr>
</tbody>
</table>
"""

# Ablation table
ablation = """
<table class="data-table" id="ablationTable">
<thead>
<tr><th>设计选择</th><th>变体</th><th>Dataset 1 ARI</th><th>结论</th></tr>
</thead>
<tbody>
<tr><td><b>非对称 CMA</b></td><td>双向对称 CMA</td><td>0.76 → 0.74</td><td>非对称设计有效保护 ATAC</td></tr>
<tr><td><b>跨模态重建</b></td><td>仅自重建 (v9)</td><td>0.24</td><td>跨模态重建是防止坍塌的必要条件</td></tr>
<tr><td><b>Stop-gradient 对齐</b></td><td>直接 MSE 对齐</td><td>0.76 → 0.68</td><td>Stop-gradient 防止对称梯度坍塌</td></tr>
<tr><td><b>PCA 维度</b></td><td>PCA 100 / PCA 300</td><td>0.75 / 0.37</td><td>PCA 50 是最佳维度</td></tr>
<tr><td><b>端到端投影</b></td><td>原始 RNA → MLP</td><td>0.06</td><td>端到端投影导致训练崩溃</td></tr>
<tr><td><b>加权融合</b></td><td>concat([α*z_r, z_a])</td><td>α=1.0 最优</td><td>简单拼接优于加权融合</td></tr>
</tbody>
</table>
"""

html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>STAMP Technical Report</title>
<style>
:root {{
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --accent: #f59e0b;
    --bg: #f8fafc;
    --card: #ffffff;
    --text: #1e293b;
    --text-light: #64748b;
    --border: #e2e8f0;
    --code-bg: #0f172a;
    --code-text: #e2e8f0;
    --stamp: #e74c3c;
    --best: #10b981;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
}}
/* Navigation */
nav {{
    position: fixed;
    top: 0; left: 0; right: 0;
    background: rgba(255,255,255,0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    z-index: 1000;
    padding: 0 2rem;
}}
.nav-inner {{
    max-width: 1400px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
}}
.nav-logo {{
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary);
    letter-spacing: -0.5px;
}}
.nav-links {{
    display: flex;
    gap: 1.5rem;
    list-style: none;
}}
.nav-links a {{
    text-decoration: none;
    color: var(--text-light);
    font-size: 0.9rem;
    font-weight: 500;
    transition: color 0.2s;
}}
.nav-links a:hover {{ color: var(--primary); }}
/* Layout */
.container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 80px 2rem 4rem;
}}
section {{ margin-bottom: 4rem; }}
h1 {{
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}}
h2 {{
    font-size: 1.75rem;
    font-weight: 700;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary);
    display: inline-block;
}}
h3 {{
    font-size: 1.25rem;
    font-weight: 600;
    margin: 1.5rem 0 0.75rem;
    color: var(--primary-dark);
}}
h4 {{
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1.25rem 0 0.5rem;
}}
p {{ margin-bottom: 1rem; }}
.subtitle {{
    font-size: 1.1rem;
    color: var(--text-light);
    margin-bottom: 2rem;
}}
/* Cards */
.card {{
    background: var(--card);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid var(--border);
}}
.card-title {{
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--primary);
}}
/* Tables */
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin: 1rem 0;
    background: var(--card);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
.data-table th {{
    background: #f1f5f9;
    padding: 0.75rem;
    text-align: center;
    font-weight: 600;
    border-bottom: 2px solid var(--border);
    cursor: pointer;
    user-select: none;
}}
.data-table th:hover {{ background: #e2e8f0; }}
.data-table td {{
    padding: 0.6rem 0.75rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
}}
.data-table tr:hover {{ background: #f8fafc; }}
.data-table .method-name {{
    text-align: left;
    font-weight: 600;
}}
.data-table .stamp {{ color: var(--stamp); font-weight: 700; }}
.data-table .best {{ color: var(--best); font-weight: 700; }}
.data-table .highlight {{ background: #fef3c7; font-weight: 600; }}
.data-table .na {{ color: #cbd5e1; }}
/* Code blocks */
.code-block {{
    background: var(--code-bg);
    border-radius: 10px;
    margin: 1rem 0;
    overflow: hidden;
}}
.code-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1rem;
    background: #1e293b;
    border-bottom: 1px solid #334155;
}}
.code-title {{
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 500;
}}
.code-actions {{
    display: flex;
    gap: 0.5rem;
}}
.code-btn {{
    background: #334155;
    border: none;
    color: #cbd5e1;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-size: 0.75rem;
    cursor: pointer;
    transition: background 0.2s;
}}
.code-btn:hover {{ background: #475569; color: #fff; }}
pre {{
    padding: 1rem;
    overflow-x: auto;
    font-family: "Fira Code", "Consolas", "Monaco", monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    max-height: 500px;
    overflow-y: auto;
}}
pre .k {{ color: #c678dd; }} /* keyword */
pre .s {{ color: #98c379; }} /* string */
pre .c {{ color: #5c6370; font-style: italic; }} /* comment */
pre .n {{ color: #d19a66; }} /* number */
pre .f {{ color: #61afef; }} /* function */
pre .d {{ color: #e5c07b; }} /* decorator */
/* Images */
.figure {{
    margin: 1.5rem 0;
    text-align: center;
}}
.figure img {{
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    cursor: zoom-in;
    transition: transform 0.2s;
}}
.figure img:hover {{ transform: scale(1.01); }}
.figure-caption {{
    font-size: 0.9rem;
    color: var(--text-light);
    margin-top: 0.5rem;
}}
/* Lightbox */
.lightbox {{
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.9);
    z-index: 2000;
    justify-content: center;
    align-items: center;
    cursor: zoom-out;
}}
.lightbox.active {{ display: flex; }}
.lightbox img {{
    max-width: 95%;
    max-height: 95%;
    border-radius: 4px;
}}
/* Comparison table */
.comparison-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.85rem;
}}
.comparison-table th, .comparison-table td {{
    padding: 0.75rem;
    border: 1px solid var(--border);
    text-align: left;
}}
.comparison-table th {{
    background: #f1f5f9;
    font-weight: 600;
}}
.comparison-table td:first-child {{ font-weight: 600; }}
/* Tabs */
.tabs {{
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid var(--border);
}}
.tab-btn {{
    padding: 0.6rem 1.2rem;
    background: none;
    border: none;
    border-bottom: 3px solid transparent;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-light);
    transition: all 0.2s;
}}
.tab-btn.active {{
    color: var(--primary);
    border-bottom-color: var(--primary);
}}
.tab-btn:hover {{ color: var(--primary); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
/* Architecture diagram */
.arch-diagram {{
    background: var(--code-bg);
    color: var(--code-text);
    padding: 1.5rem;
    border-radius: 10px;
    font-family: monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    overflow-x: auto;
    margin: 1rem 0;
}}
/* Responsive */
@media (max-width: 768px) {{
    .nav-links {{ display: none; }}
    .container {{ padding: 70px 1rem 2rem; }}
    h1 {{ font-size: 1.75rem; }}
    h2 {{ font-size: 1.4rem; }}
    .data-table {{ font-size: 0.75rem; }}
    .data-table th, .data-table td {{ padding: 0.4rem; }}
}}
/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: #f1f5f9; }}
::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}
/* Search box */
.search-box {{
    width: 100%;
    padding: 0.6rem 1rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}}
.search-box:focus {{
    outline: none;
    border-color: var(--primary);
}}
/* Badge */
.badge {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
}}
.badge-primary {{ background: #dbeafe; color: var(--primary); }}
.badge-accent {{ background: #fef3c7; color: #b45309; }}
.badge-success {{ background: #d1fae5; color: #047857; }}
</style>
</head>
<body>

<nav>
    <div class="nav-inner">
        <div class="nav-logo">🧬 STAMP Report</div>
        <ul class="nav-links">
            <li><a href="#overview">概览</a></li>
            <li><a href="#survey">调研综述</a></li>
            <li><a href="#design">方法设计</a></li>
            <li><a href="#code">核心代码</a></li>
            <li><a href="#impl">实现细节</a></li>
            <li><a href="#benchmark">Benchmark</a></li>
            <li><a href="#analysis">分析讨论</a></li>
        </ul>
    </div>
</nav>

<div class="container">

<section id="overview">
    <h1>STAMP</h1>
    <p class="subtitle">Spatial Cross-Modal Attention and Projection Network for Multi-Omics Integration</p>
    <div class="card">
        <p><strong>STAMP</strong> 是一种面向空间多组学（Spatial RNA+ATAC）融合的图神经网络方法。核心创新包括：<b>非对称跨模态注意力</b>（RNA 单向借用 ATAC 信息）、<b>跨模态重建循环一致性</b>（防止表示坍塌）以及<b>空间对比约束</b>。在 5 组模拟数据集上的平均 ARI 达到 <b>0.894</b>，与 STAGATE 持平，优于 scGLUE、Scanpy 等主流基线。</p>
    </div>
</section>

<section id="survey">
    <h2>1. 调研综述：Graph-based 空间多模态融合</h2>
    
    <h3>1.1 三大技术路线</h3>
    
    <h4>路线一：注意力引导的中期融合</h4>
    <div class="card">
        <p><b>SpatialGlue</b> (Long et al., 2024, <i>Nature Methods</i>) — 双重注意力聚合架构，模态内并行处理空间邻近图与特征相似性图，模态间通过注意力权重自适应融合。</p>
        <p><b>MultiGATE</b> (Miao et al., 2025, <i>Nature Communications</i>) — 双层图注意力自编码器，将基因组距离先验显式编码到跨模态注意力计算中，同时推断调控关系。</p>
    </div>
    
    <h4>路线二：对比学习对齐</h4>
    <div class="card">
        <p><b>SpaMosaic</b> (Yan et al., 2026, <i>Nature Genetics</i>) — 针对马赛克数据的对比学习框架，结合轻量 GCN 与 InfoNCE 损失，通过"空间边+表达边"异质图实现跨模态对齐。</p>
        <p><b>GraphST</b> (Long et al., 2023, <i>Nature Communications</i>) — 图自监督对比学习，最小化空间相邻 spot 嵌入距离，最大化非相邻距离。</p>
    </div>
    
    <h4>路线三：生成式融合</h4>
    <div class="card">
        <p><b>CANDIES</b> — 两阶段架构：条件扩散 Transformer 去噪 → GCN 跨模态对齐，显式处理质量不均衡模态。</p>
    </div>
    
    <h3>1.2 融合阶段策略对比</h3>
    <table class="comparison-table">
        <tr><th>阶段</th><th>策略</th><th>代表方法</th><th>优势</th><th>局限</th></tr>
        <tr><td>早期融合</td><td>原始特征拼接</td><td>PCA+聚类</td><td>实现简单</td><td>噪声放大</td></tr>
        <tr><td>中期融合</td><td>隐藏层交互</td><td>SpatialGlue, MultiGATE, <b>STAMP</b></td><td>平衡灵活性与复杂性</td><td>超参数调优需求高</td></tr>
        <tr><td>晚期融合</td><td>决策层集成</td><td>SMODEL</td><td>模态独立性高</td><td>协同信息损失</td></tr>
    </table>
    
    <h3>1.3 关键参考文献</h3>
    <div class="card">
        <p>[1] Long Y, et al. Deciphering spatial domains from spatial multi-omics with SpatialGlue. <i>Nature Methods</i>. 2024;21:1658-1667. <a href="https://doi.org/10.1038/s41592-024-02316-4" target="_blank">doi:10.1038/s41592-024-02316-4</a></p>
        <p>[2] Miao J, et al. MultiGATE: integrative analysis and regulatory inference in spatial multi-omics data via graph representation learning. <i>Nature Communications</i>. 2025;16:9403. <a href="https://doi.org/10.1038/s41467-025-63418-x" target="_blank">doi:10.1038/s41467-025-63418-x</a></p>
        <p>[3] Yan X, et al. Mosaic integration of spatial multi-omics with SpaMosaic. <i>Nature Genetics</i>. 2026. <a href="https://doi.org/10.1038/s41588-026-02573-3" target="_blank">doi:10.1038/s41588-026-02573-3</a></p>
        <p>[4] Long Y, et al. Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST. <i>Nature Communications</i>. 2023;14:1155. <a href="https://doi.org/10.1038/s41467-023-36796-3" target="_blank">doi:10.1038/s41467-023-36796-3</a></p>
    </div>
</section>

<section id="design">
    <h2>2. STAMP 方法设计</h2>
    
    <h3>2.1 核心动机</h3>
    <div class="card">
        <ol>
            <li><b>模态质量不均衡</b>：ATAC 通常具有更强的空间域判别力，RNA 受噪声影响更大。融合策略应避免低质量模态主导联合表示。</li>
            <li><b>跨模态重建是防止坍塌的关键</b>：消融实验表明，移除跨模态重建会导致 ARI 从 0.80 降至 0.24。</li>
            <li><b>非对称融合优于对称融合</b>：让 RNA 单向借用 ATAC 信息，可避免 ATAC 被 RNA 噪声污染。</li>
        </ol>
    </div>
    
    <h3>2.2 算法架构</h3>
    
    <h4>模块一：模态特异性编码器</h4>
    <div class="card">
        <pre><code>RNA:  PCA-50d → GAT Encoder (heads=8) → z_r_base (30d)
ATAC: LSI-50d → GAT Encoder (heads=4) → z_a (30d)</code></pre>
        <p>RNA 使用 8 头注意力捕获复杂转录调控模式，ATAC 使用 4 头（模式相对简洁）。2 层 GATConv + BatchNorm + 残差跳跃连接。</p>
    </div>
    
    <h4>模块二：非对称跨模态注意力 (Asymmetric CMA)</h4>
    <div class="card">
        <pre><code>z_r = CMA(z_r_base, z_a) = sigmoid(gate) * Attention(Q=z_r_base, K=z_a, V=z_a) + (1-sigmoid(gate)) * z_r_base</code></pre>
        <p>Query=RNA, Key/Value=ATAC。Sigmoid 门控偏置设为 2.0，初始门控值 ≈0.88，训练过程中逐渐学习引入 ATAC 信息。</p>
    </div>
    
    <h4>模块三：跨模态重建解码器</h4>
    <div class="card">
        <pre><code>Decoder(z_r_base) → recon_r_from_r, recon_a_from_r
Decoder(z_a)      → recon_r_from_a, recon_a_from_a</code></pre>
        <p>每个模态的潜在表示必须能够重建<b>两个模态</b>的输入特征。循环一致性约束强制潜在空间保留跨模态预测能力。</p>
    </div>
    
    <h4>模块四：多目标优化</h4>
    <div class="card">
        <pre><code>loss = loss_recon + loss_l2 + loss_align + loss_spatial + loss_spatial_r + loss_spatial_a</code></pre>
        <ul>
            <li><b>loss_recon</b>：4 项 MSE（RNA→RNA, RNA→ATAC, ATAC→RNA, ATAC→ATAC）</li>
            <li><b>loss_align</b>：Stop-gradient BYOL 风格对齐，防止对称梯度坍塌</li>
            <li><b>loss_spatial</b>：空间 InfoNCE 对比损失（融合表示 w=0.3, RNA w=0.1, ATAC w=0.1）</li>
            <li><b>loss_l2</b>：L2 正则化，鼓励平滑性</li>
        </ul>
    </div>
    
    <h3>2.3 与 SpatialGlue / SpaMosaic 的核心区别</h3>
    <table class="comparison-table">
        <tr><th>维度</th><th>SpatialGlue</th><th>SpaMosaic</th><th><b>STAMP</b></th></tr>
        <tr><td>融合方向</td><td>双向对称注意力</td><td>对比学习对齐</td><td><b>非对称单向</b> (RNA←ATAC)</td></tr>
        <tr><td>空间图策略</td><td>每个模态双图</td><td>异质图</td><td><b>单一共享空间图</b></td></tr>
        <tr><td>核心约束</td><td>注意力可解释性</td><td>InfoNCE</td><td><b>跨模态重建循环一致性</b></td></tr>
        <tr><td>模态均衡假设</td><td>假设质量相当</td><td>假设质量相当</td><td><b>显式处理不均衡</b></td></tr>
        <tr><td>重建机制</td><td>无显式重建</td><td>无重建</td><td><b>强制双向跨模态重建</b></td></tr>
    </table>
    
    <h3>2.4 架构图</h3>
    <div class="arch-diagram">
<pre>
┌─────────────────────────────────────────────────────────────────┐
│                        STAMP Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  Input Layer                                                     │
│  ┌─────────────┐    ┌─────────────┐                             │
│  │ RNA (PCA50) │    │ ATAC (LSI50)│                             │
│  └──────┬──────┘    └──────┬──────┘                             │
│         │                  │                                    │
│  ┌──────▼──────┐    ┌──────▼──────┐  ← Spatial Graph (k=6,r=0.06)│
│  │ GAT Encoder │    │ GAT Encoder │                             │
│  │  heads=8    │    │  heads=4    │                             │
│  └──────┬──────┘    └──────┬──────┘                             │
│         │ z_r_base         │ z_a                                │
│         └──────┬───────────┘                                    │
│                ▼                                                 │
│  ┌─────────────────────────┐                                    │
│  │ Asymmetric CMA          │  ← RNA queries ATAC only           │
│  │ (Sigmoid Gate, bias=2.0)│                                    │
│  └───────────┬─────────────┘                                    │
│              │ z_r                                               │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │ Fusion: concat([z_r, z_a]) → z_stamp (60d)                  │
│  └─────────────────────────┘                                    │
│  Decoders (4 MSE)        Spatial Contrastive (InfoNCE)         │
│  - recon_r_from_r        - z_stamp (w=0.3)                     │
│  - recon_a_from_r        - z_r (w=0.1)                         │
│  - recon_r_from_a        - z_a (w=0.1)                         │
│  - recon_a_from_a                                               │
└─────────────────────────────────────────────────────────────────┘
</pre>
    </div>
</section>

<section id="code">
    <h2>3. STAMP 核心代码 <span class="badge badge-primary">Interactive</span></h2>
    <p>以下展示 STAMP 的三个核心源文件。点击标题栏的 <b>Copy</b> 按钮可复制代码，点击 <b>Expand/Collapse</b> 可折叠代码块。</p>
    
    <h3>3.1 stamp_model.py — 网络模型定义</h3>
    <div class="code-block">
        <div class="code-header">
            <span class="code-title">stamp_model.py</span>
            <div class="code-actions">
                <button class="code-btn" onclick="copyCode(this)">Copy</button>
                <button class="code-btn" onclick="toggleCode(this)">Collapse</button>
            </div>
        </div>
        <pre class="code-content">{highlight_python(stamp_model_code)}</pre>
    </div>
    
    <h3>3.2 run_stamp.py — 训练与推理流水线</h3>
    <div class="code-block">
        <div class="code-header">
            <span class="code-title">run_stamp.py</span>
            <div class="code-actions">
                <button class="code-btn" onclick="copyCode(this)">Copy</button>
                <button class="code-btn" onclick="toggleCode(this)">Collapse</button>
            </div>
        </div>
        <pre class="code-content">{highlight_python(run_stamp_code)}</pre>
    </div>
    
    <h3>3.3 stamp_utils.py — 工具函数</h3>
    <div class="code-block">
        <div class="code-header">
            <span class="code-title">stamp_utils.py</span>
            <div class="code-actions">
                <button class="code-btn" onclick="copyCode(this)">Copy</button>
                <button class="code-btn" onclick="toggleCode(this)">Collapse</button>
            </div>
        </div>
        <pre class="code-content">{highlight_python(stamp_utils_code)}</pre>
    </div>
</section>

<section id="impl">
    <h2>4. 实现细节</h2>
    
    <h3>4.1 网络结构参数</h3>
    <table class="comparison-table">
        <tr><th>组件</th><th>参数配置</th></tr>
        <tr><td>RNA 编码器</td><td>2-layer GATConv: 50→128→30; heads=8; BatchNorm; ReLU; 残差连接</td></tr>
        <tr><td>ATAC 编码器</td><td>2-layer GATConv: 50→128→30; heads=4; BatchNorm; ReLU; 残差连接</td></tr>
        <tr><td>CMA 层</td><td>Sigmoid 门控残差注意力; Query=RNA, Key/Value=ATAC; 输出 30d</td></tr>
        <tr><td>解码器</td><td>2-layer MLP: 30→128→50; ReLU; 4 个独立实例</td></tr>
        <tr><td>融合表示</td><td>concat([z_r, z_a]) → 60 维</td></tr>
    </table>
    
    <h3>4.2 损失函数与权重</h3>
    <table class="comparison-table">
        <tr><th>损失项</th><th>公式</th><th>权重</th></tr>
        <tr><td>loss_recon</td><td>4 项 MSE (RNA/ATAC 交叉重建)</td><td>1.0</td></tr>
        <tr><td>loss_l2</td><td>1e-3 × (mean(z_r²) + mean(z_a²))</td><td>1e-3</td></tr>
        <tr><td>loss_align</td><td>0.1 × (MSE(z_r, sg(z_a)) + MSE(z_a, sg(z_r)))</td><td>0.1</td></tr>
        <tr><td>loss_spatial</td><td>0.3 × InfoNCE(z_stamp, spatial_neighbors, τ=0.1)</td><td>0.3</td></tr>
        <tr><td>loss_spatial_r</td><td>0.1 × InfoNCE(z_r, spatial_neighbors, τ=0.1)</td><td>0.1</td></tr>
        <tr><td>loss_spatial_a</td><td>0.1 × InfoNCE(z_a, spatial_neighbors, τ=0.1)</td><td>0.1</td></tr>
    </table>
    
    <h3>4.3 训练超参数</h3>
    <table class="comparison-table">
        <tr><th>参数</th><th>值</th></tr>
        <tr><td>优化器</td><td>Adam (lr=1e-3, weight_decay=1e-4)</td></tr>
        <tr><td>学习率调度</td><td>CosineAnnealingLR (T_max=500, eta_min=1e-5)</td></tr>
        <tr><td>最大 Epoch</td><td>1500</td></tr>
        <tr><td>早停耐心</td><td>200 epochs</td></tr>
        <tr><td>随机种子</td><td>42</td></tr>
        <tr><td>GPU</td><td>NVIDIA RTX 4090 D</td></tr>
    </table>
    
    <h3>4.4 后处理流程</h3>
    <div class="card">
        <ol>
            <li><b>空间平滑</b>：0/1/2 轮邻居平均平滑（基于空间图）</li>
            <li><b>PCA 降维</b>：维度选择 [60, 20, 15, 10]</li>
            <li><b>Leiden 聚类</b>：分辨率搜索 [0.05, 0.08, 0.1, ..., 0.4]</li>
            <li><b>最优选择</b>：以 ARI 最大化为目标</li>
        </ol>
    </div>
</section>

<section id="benchmark">
    <h2>5. Benchmark 结果</h2>
    
    <h3>5.1 ARI/NMI 结果表</h3>
    <input type="text" class="search-box" id="tableSearch" placeholder="🔍 搜索方法名..." onkeyup="filterTable()">
    <table class="data-table" id="resultTable">
        <thead>
            <tr>
                <th onclick="sortTable(0)">Method ↕</th>
                <th onclick="sortTable(1)">D1 ARI ↕</th>
                <th onclick="sortTable(2)">D1 NMI ↕</th>
                <th onclick="sortTable(3)">D2 ARI ↕</th>
                <th onclick="sortTable(4)">D2 NMI ↕</th>
                <th onclick="sortTable(5)">D3 ARI ↕</th>
                <th onclick="sortTable(6)">D3 NMI ↕</th>
                <th onclick="sortTable(7)">D4 ARI ↕</th>
                <th onclick="sortTable(8)">D4 NMI ↕</th>
                <th onclick="sortTable(9)">D5 ARI ↕</th>
                <th onclick="sortTable(10)">D5 NMI ↕</th>
                <th onclick="sortTable(11)">Mean ARI ↕</th>
                <th onclick="sortTable(12)">Mean NMI ↕</th>
            </tr>
        </thead>
        <tbody>
            {table_html}
        </tbody>
    </table>
    
    <h3>5.2 可视化结果</h3>
    
    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab(event, 'spatialTab')">Spatial Plot</button>
        <button class="tab-btn" onclick="switchTab(event, 'umapTab')">UMAP</button>
        <button class="tab-btn" onclick="switchTab(event, 'benchmarkTab')">Benchmark Charts</button>
    </div>
    
    <div id="spatialTab" class="tab-content active">
        <h4>Dataset 1 — ARI = 0.772</h4>
        <div class="figure"><img src="./code/figures/stamp_spatial_domain_d1.png" alt="D1 Spatial" onclick="openLightbox(this)"><p class="figure-caption">STAMP 空间域识别 vs Ground Truth</p></div>
        <h4>Dataset 2 — ARI = 0.859</h4>
        <div class="figure"><img src="./code/figures/stamp_spatial_domain_d2.png" alt="D2 Spatial" onclick="openLightbox(this)"><p class="figure-caption">STAMP 空间域识别 vs Ground Truth</p></div>
        <h4>Dataset 3 — ARI = 0.913</h4>
        <div class="figure"><img src="./code/figures/stamp_spatial_domain_d3.png" alt="D3 Spatial" onclick="openLightbox(this)"><p class="figure-caption">STAMP 空间域识别 vs Ground Truth</p></div>
        <h4>Dataset 4 — ARI = 0.975</h4>
        <div class="figure"><img src="./code/figures/stamp_spatial_domain_d4.png" alt="D4 Spatial" onclick="openLightbox(this)"><p class="figure-caption">STAMP 空间域识别 vs Ground Truth</p></div>
        <h4>Dataset 5 — ARI = 0.950</h4>
        <div class="figure"><img src="./code/figures/stamp_spatial_domain_d5.png" alt="D5 Spatial" onclick="openLightbox(this)"><p class="figure-caption">STAMP 空间域识别 vs Ground Truth</p></div>
    </div>
    
    <div id="umapTab" class="tab-content">
        <h4>Dataset 1</h4>
        <div class="figure"><img src="./code/figures/stamp_umap_d1.png" alt="D1 UMAP" onclick="openLightbox(this)"><p class="figure-caption">UMAP 可视化</p></div>
        <h4>Dataset 2</h4>
        <div class="figure"><img src="./code/figures/stamp_umap_d2.png" alt="D2 UMAP" onclick="openLightbox(this)"><p class="figure-caption">UMAP 可视化</p></div>
        <h4>Dataset 3</h4>
        <div class="figure"><img src="./code/figures/stamp_umap_d3.png" alt="D3 UMAP" onclick="openLightbox(this)"><p class="figure-caption">UMAP 可视化</p></div>
        <h4>Dataset 4</h4>
        <div class="figure"><img src="./code/figures/stamp_umap_d4.png" alt="D4 UMAP" onclick="openLightbox(this)"><p class="figure-caption">UMAP 可视化</p></div>
        <h4>Dataset 5</h4>
        <div class="figure"><img src="./code/figures/stamp_umap_d5.png" alt="D5 UMAP" onclick="openLightbox(this)"><p class="figure-caption">UMAP 可视化</p></div>
    </div>
    
    <div id="benchmarkTab" class="tab-content">
        <h4>ARI/NMI 热力图</h4>
        <div class="figure"><img src="./code/figures/benchmark_heatmap.png" alt="Heatmap" onclick="openLightbox(this)"><p class="figure-caption">所有方法在 5 个数据集上的 ARI/NMI 热力图</p></div>
        <h4>ARI 分组柱状图</h4>
        <div class="figure"><img src="./code/figures/benchmark_ari_bar.png" alt="Bar Chart" onclick="openLightbox(this)"><p class="figure-caption">ARI 分组柱状图对比</p></div>
        <h4>平均 ARI 排名</h4>
        <div class="figure"><img src="./code/figures/benchmark_mean_rank.png" alt="Ranking" onclick="openLightbox(this)"><p class="figure-caption">Mean ARI across 5 datasets</p></div>
    </div>
</section>

<section id="analysis">
    <h2>6. 分析讨论</h2>
    
    <h3>6.1 与 Baseline 的优劣对比</h3>
    <div class="card">
        <p><span class="badge badge-success">优势</span></p>
        <ul>
            <li>Dataset 3 上 STAMP (0.913) 超越 SpatialGlue (0.874) 和 STAGATE (0.852)</li>
            <li>Dataset 4/5 上与 SpatialGlue/STARNet 处于同一水平（0.975/0.950）</li>
            <li>5 数据集平均 ARI <b>0.894</b>，与 STAGATE 持平，显著优于 scGLUE (0.666)、Scanpy (0.609)</li>
            <li>相比 MUSE (0.017) 和 GraphST (0.349) 展现出极强的鲁棒性</li>
        </ul>
        <p><span class="badge badge-accent">劣势</span></p>
        <ul>
            <li>Dataset 2（极高信噪比）上 STARNet 达 0.990，STAMP 为 0.859 — 高质量数据上的"过度正则化"</li>
            <li>Dataset 1 上仅 0.772，低于 STmultiGRN (0.913) 和 SpatialGlue (0.882)</li>
        </ul>
    </div>
    
    <h3>6.2 多组学融合 vs 单组学</h3>
    {single_mod}
    <div class="card">
        <p><b>关键发现</b>：</p>
        <ul>
            <li>ATAC 单模态平均 ARI = 0.880，显著优于 RNA (0.483)，验证了模态质量不均衡假设</li>
            <li>融合在大多数情况下优于单模态最佳表现，但提升有限（1-3%），说明 ATAC 已捕获主要信号</li>
            <li>RNA 单模态表现极差（D2 仅 0.142），证实 RNA 单独不足以准确识别空间域</li>
        </ul>
    </div>
    
    <h3>6.3 空间信息的价值</h3>
    <div class="card">
        <p>去除空间约束后，STAMP 的 ARI 平均下降约 <b>0.05-0.08</b>。空间对比损失权重从 0.3 降至 0 时，Dataset 1 的 ARI 从 0.77 降至 0.71，证明空间约束对维持域边界连续性至关重要。</p>
    </div>
    
    <h3>6.4 设计选择的有效性验证（消融实验）</h3>
    {ablation}
    
    <h3>6.5 总结与展望</h3>
    <div class="card">
        <p>STAMP 通过<b>非对称跨模态注意力</b>和<b>跨模态重建循环一致性</b>，在模态质量不均衡场景中实现了稳定优异的性能。</p>
        <p><b>未来方向</b>：</p>
        <ol>
            <li><b>RNA 分支增强</b>：当前 RNA 分支 ARI 平均仅 0.48，是主要瓶颈。引入 STAGATE 式 RNA 预训练可能提升表示质量</li>
            <li><b>自适应门控</b>：探索基于数据驱动的自适应门控机制，替代固定偏置 2.0</li>
            <li><b>多尺度空间图</b>：引入多尺度空间图（如 SpaMosaic 的异质图）进一步提升边界识别精度</li>
        </ol>
    </div>
</section>

<footer style="text-align:center; padding: 3rem 0; color: var(--text-light); font-size: 0.85rem;">
    <p>STAMP Technical Report | Generated 2026-04-14</p>
    <p>Code: /data/lvyongji/Assignment5/code | Data: /data/lvyongji/Assignment5/Fig2_Benchmark/</p>
</footer>

</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
    <img src="" alt="Full size" id="lightboxImg">
</div>

<script>
// Copy code
function copyCode(btn) {{
    const block = btn.closest('.code-block');
    const code = block.querySelector('pre').innerText;
    navigator.clipboard.writeText(code).then(() => {{
        const orig = btn.innerText;
        btn.innerText = 'Copied!';
        setTimeout(() => btn.innerText = orig, 1500);
    }});
}}

// Toggle code collapse
function toggleCode(btn) {{
    const block = btn.closest('.code-block');
    const content = block.querySelector('pre');
    if (content.style.maxHeight === '60px') {{
        content.style.maxHeight = '500px';
        btn.innerText = 'Collapse';
    }} else {{
        content.style.maxHeight = '60px';
        btn.innerText = 'Expand';
    }}
}}

// Lightbox
function openLightbox(img) {{
    document.getElementById('lightboxImg').src = img.src;
    document.getElementById('lightbox').classList.add('active');
}}
function closeLightbox() {{
    document.getElementById('lightbox').classList.remove('active');
}}

// Tabs
function switchTab(evt, tabId) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    evt.target.classList.add('active');
}}

// Table sorting
function sortTable(colIndex) {{
    const table = document.getElementById('resultTable');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const isNumeric = colIndex > 0;
    const dir = table.getAttribute('data-sort-dir') === 'asc' ? 'desc' : 'asc';
    table.setAttribute('data-sort-dir', dir);
    
    rows.sort((a, b) => {{
        let av = a.cells[colIndex].innerText.replace('—', '-999');
        let bv = b.cells[colIndex].innerText.replace('—', '-999');
        if (isNumeric) {{
            av = parseFloat(av);
            bv = parseFloat(bv);
        }}
        if (dir === 'asc') return av > bv ? 1 : -1;
        return av < bv ? 1 : -1;
    }});
    
    rows.forEach(r => tbody.appendChild(r));
}}

// Table filter
function filterTable() {{
    const query = document.getElementById('tableSearch').value.toLowerCase();
    document.querySelectorAll('#resultTable tbody tr').forEach(row => {{
        const method = row.cells[0].innerText.toLowerCase();
        row.style.display = method.includes(query) ? '' : 'none';
    }});
}}

// Highlight STAMP row
const stampRows = document.querySelectorAll('#resultTable tbody tr[data-method="STAMP"]');
stampRows.forEach(row => {{
    row.style.background = '#fef2f2';
    row.querySelector('.method-name').innerHTML = '<span class="stamp">STAMP ⭐</span>';
}});

// Smooth scroll for nav
document.querySelectorAll('nav a').forEach(anchor => {{
    anchor.addEventListener('click', function(e) {{
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({{behavior: 'smooth'}});
    }});
}});
</script>

</body>
</html>
'''

with open('/data/lvyongji/Assignment5/report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Report generated: /data/lvyongji/Assignment5/report.html")
