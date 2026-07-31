import json
import html

# v7a benchmark results
v7a_results = {
    '1': {'ARI': 0.8151, 'NMI': 0.7544},
    '2': {'ARI': 0.8633, 'NMI': 0.8015},
    '3': {'ARI': 0.9153, 'NMI': 0.8937},
    '4': {'ARI': 0.9614, 'NMI': 0.9434},
    '5': {'ARI': 0.9619, 'NMI': 0.9431},
}

# Load baseline results
with open('/data/lvyongji/Assignment5/code/benchmark_results.json') as f:
    baseline = json.load(f)

# Read code files
with open('/data/lvyongji/Assignment5/code/stamp_model_v7a.py') as f:
    stamp_model_code = f.read()
with open('/data/lvyongji/Assignment5/code/run_stamp_v7a.py') as f:
    run_stamp_code = f.read()

def esc(s):
    return html.escape(s)

def highlight_python(code):
    import re
    keywords = ['def', 'class', 'return', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'in', 'not', 'and', 'or', 'True', 'False', 'None', 'self', 'super', 'pass', 'break', 'continue', 'try', 'except', 'finally', 'with', 'as', 'yield', 'lambda', 'assert', 'del', 'global', 'nonlocal', 'raise']
    code = esc(code)
    code = re.sub(r'(#.*?)$', r'<span class="c">\1</span>', code, flags=re.MULTILINE)
    code = re.sub(r'(""".*?""")', r'<span class="s">\1</span>', code, flags=re.DOTALL)
    code = re.sub(r"('''.*?''')", r'<span class="s">\1</span>', code, flags=re.DOTALL)
    code = re.sub(r'(".*?")', r'<span class="s">\1</span>', code)
    code = re.sub(r"('.*?')", r'<span class="s">\1</span>', code)
    code = re.sub(r'\b(\d+\.?\d*)\b', r'<span class="n">\1</span>', code)
    for kw in keywords:
        code = re.sub(r'\b(' + kw + r')\b', r'<span class="k">\1</span>', code)
    code = re.sub(r'(@\w+)', r'<span class="d">\1</span>', code)
    code = re.sub(r'(\w+)(\()', r'<span class="f">\1</span>\2', code)
    return code

# Build result table with v7a added
methods = sorted(set(m for r in baseline.values() for m in r.keys()))
all_methods = methods + ['STAMPv7']
datasets = ['1','2','3','4','5']

table_rows = []
for m in all_methods:
    row = f'<tr data-method="{m}">'
    row += f'<td class="method-name">{m}</td>'
    mean_ari = 0
    mean_nmi = 0
    count = 0
    for d in datasets:
        if m == 'STAMPv7':
            ari = v7a_results[d]['ARI']
            nmi = v7a_results[d]['NMI']
        elif m in baseline[d]:
            ari = baseline[d][m]['ARI']
            nmi = baseline[d][m]['NMI']
        else:
            ari = None
            nmi = None
        if ari is not None:
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

# v6.1 vs v7a comparison table
comparison_table = """
<table class="data-table">
<thead><tr><th>Dataset</th><th>v6.1 ARI</th><th>v7a ARI</th><th>Δ ARI</th><th>v6.1 NMI</th><th>v7a NMI</th><th>Δ NMI</th></tr></thead>
<tbody>
<tr><td>D1</td><td>0.7717</td><td class="highlight">0.8151</td><td class="best">+0.0434</td><td>0.6969</td><td class="highlight">0.7544</td><td class="best">+0.0575</td></tr>
<tr><td>D2</td><td>0.8590</td><td class="highlight">0.8633</td><td>+0.0043</td><td>0.8002</td><td class="highlight">0.8015</td><td>+0.0013</td></tr>
<tr><td>D3</td><td>0.9128</td><td class="highlight">0.9153</td><td>+0.0025</td><td>0.8977</td><td>0.8937</td><td>-0.0040</td></tr>
<tr><td>D4</td><td class="highlight">0.9751</td><td>0.9614</td><td class="stamp">-0.0137</td><td class="highlight">0.9638</td><td>0.9434</td><td class="stamp">-0.0204</td></tr>
<tr><td>D5</td><td>0.9499</td><td class="highlight">0.9619</td><td>+0.0120</td><td>0.9298</td><td class="highlight">0.9431</td><td>+0.0133</td></tr>
<tr style="background:#f0f9ff;font-weight:600"><td>Mean</td><td>0.8937</td><td class="highlight">0.9034</td><td class="best">+0.0097</td><td>0.8580</td><td class="highlight">0.8672</td><td>+0.0092</td></tr>
</tbody>
</table>
"""

# Single modality breakdown
single_mod = """
<table class="data-table" id="singleModTable">
<thead>
<tr><th>Dataset</th><th>v6.1 z_r</th><th>v7a z_r</th><th>v6.1 z_a</th><th>v7a z_a</th><th>v6.1 z_stamp</th><th>v7a z_stamp</th></tr>
</thead>
<tbody>
<tr><td>D1</td><td>0.5191</td><td>0.3489</td><td>0.7478</td><td class="highlight">0.8274</td><td>0.7717</td><td class="highlight">0.8151</td></tr>
<tr><td>D2</td><td>0.1417</td><td>0.2161</td><td>0.8699</td><td class="highlight">0.8927</td><td>0.8590</td><td class="highlight">0.8633</td></tr>
<tr><td>D3</td><td>0.6374</td><td>0.4144</td><td>0.8835</td><td class="highlight">0.9185</td><td>0.9128</td><td class="highlight">0.9153</td></tr>
<tr><td>D4</td><td>0.6195</td><td>0.3528</td><td>0.9619</td><td>0.9617</td><td class="highlight">0.9751</td><td>0.9614</td></tr>
<tr><td>D5</td><td>0.4973</td><td>0.4327</td><td>0.9358</td><td class="highlight">0.9597</td><td>0.9499</td><td class="highlight">0.9619</td></tr>
</tbody>
</table>
"""

html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>STAMP v7a Technical Report</title>
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
    --v61: #f39c12;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
}}
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
.nav-logo {{ font-size: 1.25rem; font-weight: 700; color: var(--primary); letter-spacing: -0.5px; }}
.nav-links {{ display: flex; gap: 1.5rem; list-style: none; }}
.nav-links a {{ text-decoration: none; color: var(--text-light); font-size: 0.9rem; font-weight: 500; transition: color 0.2s; }}
.nav-links a:hover {{ color: var(--primary); }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 80px 2rem 4rem; }}
section {{ margin-bottom: 4rem; }}
h1 {{ font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: -1px; }}
h2 {{ font-size: 1.75rem; font-weight: 700; margin: 2.5rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--primary); display: inline-block; }}
h3 {{ font-size: 1.25rem; font-weight: 600; margin: 1.5rem 0 0.75rem; color: var(--primary-dark); }}
h4 {{ font-size: 1.1rem; font-weight: 600; margin: 1.25rem 0 0.5rem; }}
p {{ margin-bottom: 1rem; }}
.subtitle {{ font-size: 1.1rem; color: var(--text-light); margin-bottom: 2rem; }}
.card {{
    background: var(--card);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    border: 1px solid var(--border);
}}
.card-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 0.75rem; color: var(--primary); }}
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
.data-table .method-name {{ text-align: left; font-weight: 600; }}
.data-table .stamp {{ color: var(--stamp); font-weight: 700; }}
.data-table .best {{ color: var(--best); font-weight: 700; }}
.data-table .highlight {{ background: #fef3c7; font-weight: 600; }}
.data-table .na {{ color: #cbd5e1; }}
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
.code-title {{ font-size: 0.85rem; color: #94a3b8; font-weight: 500; }}
.code-actions {{ display: flex; gap: 0.5rem; }}
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
pre .k {{ color: #c678dd; }}
pre .s {{ color: #98c379; }}
pre .c {{ color: #5c6370; font-style: italic; }}
pre .n {{ color: #d19a66; }}
pre .f {{ color: #61afef; }}
pre .d {{ color: #e5c07b; }}
.figure {{ margin: 1.5rem 0; text-align: center; }}
.figure img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); cursor: zoom-in; transition: transform 0.2s; }}
.figure img:hover {{ transform: scale(1.01); }}
.figure-caption {{ font-size: 0.9rem; color: var(--text-light); margin-top: 0.5rem; }}
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
.lightbox img {{ max-width: 95%; max-height: 95%; border-radius: 4px; }}
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
.comparison-table th {{ background: #f1f5f9; font-weight: 600; }}
.comparison-table td:first-child {{ font-weight: 600; }}
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
.tab-btn.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
.tab-btn:hover {{ color: var(--primary); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
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
@media (max-width: 768px) {{
    .nav-links {{ display: none; }}
    .container {{ padding: 70px 1rem 2rem; }}
    h1 {{ font-size: 1.75rem; }}
    h2 {{ font-size: 1.4rem; }}
    .data-table {{ font-size: 0.75rem; }}
    .data-table th, .data-table td {{ padding: 0.4rem; }}
}}
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: #f1f5f9; }}
::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}
.search-box {{
    width: 100%;
    padding: 0.6rem 1rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}}
.search-box:focus {{ outline: none; border-color: var(--primary); }}
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
.version-banner {{
    background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
    border-left: 4px solid var(--primary);
    padding: 1rem 1.5rem;
    border-radius: 8px;
    margin-bottom: 2rem;
}}
.version-banner h3 {{ margin: 0; color: var(--primary-dark); }}
</style>
</head>
<body>

<nav>
    <div class="nav-inner">
        <div class="nav-logo">🧬 STAMP v7a Report</div>
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
    <h1>STAMP v7a</h1>
    <p class="subtitle">Spatial Cross-Modal Attention with Dual-Graph Encoding</p>
    <div class="version-banner">
        <h3>Version History</h3>
        <p><b>v5</b> → ARI=0.777 (baseline) | <b>v6.1</b> → ARI=0.894 (stable) | <b>v7a</b> → ARI=0.903 (dual-graph)</p>
    </div>
    <div class="card">
        <p><strong>STAMP v7a</strong> 在 v6.1 的基础上引入<b>双图并行编码器</b>（空间图 + 特征相似性图），将平均 ARI 从 <b>0.894</b> 提升至 <b>0.903</b>，已超越 STAGATE（0.894），接近 SpatialGlue（0.936）。核心设计包括：非对称跨模态注意力、跨模态重建循环一致性、以及新增的双图 GAT 编码器。</p>
    </div>
</section>

<section id="survey">
    <h2>1. 调研综述：Graph-based 空间多模态融合</h2>
    
    <h3>1.1 三大技术路线</h3>
    
    <h4>路线一：注意力引导的中期融合</h4>
    <div class="card">
        <p><b>SpatialGlue</b> (Long et al., 2024, <i>Nature Methods</i>) — 双重注意力聚合架构，模态内并行处理空间邻近图与特征相似性图。</p>
        <p><b>MultiGATE</b> (Miao et al., 2025, <i>Nature Communications</i>) — 双层图注意力自编码器，将基因组距离先验编码到跨模态注意力。</p>
    </div>
    
    <h4>路线二：对比学习对齐</h4>
    <div class="card">
        <p><b>SpaMosaic</b> (Yan et al., 2026, <i>Nature Genetics</i>) — 马赛克数据对比学习框架，结合 GCN 与 InfoNCE。</p>
        <p><b>GraphST</b> (Long et al., 2023, <i>Nature Communications</i>) — 图自监督对比学习。</p>
    </div>
    
    <h3>1.2 关键参考文献</h3>
    <div class="card">
        <p>[1] Long Y, et al. Deciphering spatial domains from spatial multi-omics with SpatialGlue. <i>Nature Methods</i>. 2024;21:1658-1667.</p>
        <p>[2] Miao J, et al. MultiGATE: integrative analysis and regulatory inference in spatial multi-omics data via graph representation learning. <i>Nature Communications</i>. 2025;16:9403.</p>
        <p>[3] Yan X, et al. Mosaic integration of spatial multi-omics with SpaMosaic. <i>Nature Genetics</i>. 2026.</p>
        <p>[4] Long Y, et al. Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST. <i>Nature Communications</i>. 2023;14:1155.</p>
    </div>
</section>

<section id="design">
    <h2>2. STAMP v7a 方法设计</h2>
    
    <h3>2.1 核心动机与改进历程</h3>
    <div class="card">
        <ul>
            <li><b>v5 基线</b>：传统对称 CMA + 空间对比，ARI=0.777（D1）</li>
            <li><b>v6.1 稳定版</b>：非对称 CMA + 跨模态重建，Mean ARI=0.894</li>
            <li><b>v7a 改进版</b>：引入<b>双图并行编码器</b>（空间图 + 特征相似性图），Mean ARI=<b>0.903</b></li>
        </ul>
        <p>v7a 的设计基于两个观察：<b>空间图 alone 不足</b>（空间邻近不一定特征相似），以及<b>双图已被验证有效</b>（SpatialGlue 的核心优势）。</p>
    </div>
    
    <h3>2.2 双图 GAT 编码器（DualGATEncoder）—— v7a 核心创新</h3>
    <div class="card">
        <pre><code>RNA:  PCA-50d  →  DualGATEncoder(spatial_graph, feature_graph)  →  z_r_base (30d)
ATAC: LSI-50d  →  DualGATEncoder(spatial_graph, feature_graph)  →  z_a (30d)</code></pre>
        <ul>
            <li><b>空间图</b>：基于物理坐标的 kNN（k=6, radius=0.06）</li>
            <li><b>特征图</b>：基于 PCA/LSI 低维表示的 kNN（k=15）</li>
            <li><b>可学习融合</b>：每层后用 sigmoid(alpha) 加权融合，初始 alpha=0.85 偏向空间图</li>
        </ul>
        <p><b>与 SpatialGlue 的区别</b>：SpatialGlue 使用完全独立的双编码器；STAMPv7a 使用<b>共享权重的 GAT 层</b>分别处理两个图，参数更高效。</p>
    </div>
    
    <h3>2.3 其他核心模块</h3>
    <div class="card">
        <p><b>非对称 CMA</b>：仅 RNA 查询 ATAC，Sigmoid 门控偏置 2.0，初始保留 88% 原始 RNA 信号</p>
        <p><b>跨模态重建</b>：每个模态潜在表示必须重建两个模态，防止坍塌的必要条件</p>
        <p><b>多目标优化</b>：重建损失 + L2 + Stop-gradient 对齐 + 空间 InfoNCE（融合 w=0.3, RNA/ATAC 各 w=0.1）</p>
    </div>
    
    <h3>2.4 与 SpatialGlue / SpaMosaic 的核心区别</h3>
    <table class="comparison-table">
        <tr><th>维度</th><th>SpatialGlue</th><th>SpaMosaic</th><th><b>STAMP v7a</b></th></tr>
        <tr><td>融合方向</td><td>双向对称注意力</td><td>对比学习对齐</td><td><b>非对称单向</b> (RNA←ATAC)</td></tr>
        <tr><td>双图策略</td><td>双独立编码器</td><td>异质图</td><td><b>双图共享权重编码器</b></td></tr>
        <tr><td>核心约束</td><td>注意力可解释性</td><td>InfoNCE</td><td><b>跨模态重建循环一致性</b></td></tr>
        <tr><td>模态均衡假设</td><td>假设质量相当</td><td>假设质量相当</td><td><b>显式保护高质量模态（ATAC）</b></td></tr>
    </table>
    
    <h3>2.5 架构图</h3>
    <div class="arch-diagram">
<pre>
┌─────────────────────────────────────────────────────────────────┐
│                    STAMP v7a Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  Input Layer                                                     │
│  ┌─────────────┐    ┌─────────────┐                             │
│  │ RNA (PCA50) │    │ ATAC (LSI50)│                             │
│  └──────┬──────┘    └──────┬──────┘                             │
│         │                  │                                    │
│  ┌──────▼──────────────────▼──────┐                             │
│  │ DualGATEncoder                  │                             │
│  │  ├─ Spatial Graph (k=6,r=0.06) │ ← shared GAT weights        │
│  │  └─ Feature Graph (k=15)       │    fused via learnable α    │
│  └──────┬──────────────────┬──────┘                             │
│         │ z_r_base         │ z_a                                │
│         └──────┬───────────┘                                    │
│                ▼                                                 │
│  ┌─────────────────────────┐                                    │
│  │ Asymmetric CMA          │  ← RNA queries ATAC only           │
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
    <h2>3. STAMP v7a 核心代码 <span class="badge badge-primary">Interactive</span></h2>
    
    <h3>3.1 stamp_model_v7a.py — 双图模型定义</h3>
    <div class="code-block">
        <div class="code-header">
            <span class="code-title">stamp_model_v7a.py</span>
            <div class="code-actions">
                <button class="code-btn" onclick="copyCode(this)">Copy</button>
                <button class="code-btn" onclick="toggleCode(this)">Collapse</button>
            </div>
        </div>
        <pre class="code-content">{highlight_python(stamp_model_code)}</pre>
    </div>
    
    <h3>3.2 run_stamp_v7a.py — 训练与推理流水线</h3>
    <div class="code-block">
        <div class="code-header">
            <span class="code-title">run_stamp_v7a.py</span>
            <div class="code-actions">
                <button class="code-btn" onclick="copyCode(this)">Copy</button>
                <button class="code-btn" onclick="toggleCode(this)">Collapse</button>
            </div>
        </div>
        <pre class="code-content">{highlight_python(run_stamp_code)}</pre>
    </div>
</section>

<section id="impl">
    <h2>4. 实现细节</h2>
    
    <h3>4.1 网络结构参数</h3>
    <table class="comparison-table">
        <tr><th>组件</th><th>参数配置</th></tr>
        <tr><td>DualGATEncoder (RNA)</td><td>2-layer GATConv: 50→128→30; heads=8; BatchNorm; ReLU; 残差连接; 双图可学习融合</td></tr>
        <tr><td>DualGATEncoder (ATAC)</td><td>2-layer GATConv: 50→128→30; heads=4; BatchNorm; ReLU; 残差连接; 双图可学习融合</td></tr>
        <tr><td>CMA 层</td><td>Sigmoid 门控残差注意力; Query=RNA, Key/Value=ATAC</td></tr>
        <tr><td>解码器</td><td>2-layer MLP: 30→128→50; ReLU; 4 个独立实例</td></tr>
    </table>
    
    <h3>4.2 图构建参数</h3>
    <table class="comparison-table">
        <tr><th>图类型</th><th>构建方式</th><th>参数</th></tr>
        <tr><td>空间图</td><td>归一化坐标 kNN + 距离阈值</td><td>k=6, radius=0.06</td></tr>
        <tr><td>RNA 特征图</td><td>PCA-50d 空间 kNN</td><td>k=15, connectivity</td></tr>
        <tr><td>ATAC 特征图</td><td>LSI-50d 空间 kNN</td><td>k=15, connectivity</td></tr>
    </table>
    
    <h3>4.3 训练超参数</h3>
    <table class="comparison-table">
        <tr><th>参数</th><th>值</th></tr>
        <tr><td>优化器</td><td>Adam (lr=1e-3, weight_decay=1e-4)</td></tr>
        <tr><td>学习率调度</td><td>CosineAnnealingLR (T_max=500, eta_min=1e-5)</td></tr>
        <tr><td>最大 Epoch</td><td>1500</td></tr>
        <tr><td>早停耐心</td><td>200 epochs</td></tr>
        <tr><td>随机种子</td><td>42</td></tr>
    </table>
</section>

<section id="benchmark">
    <h2>5. Benchmark 结果</h2>
    
    <h3>5.1 v6.1 vs v7a 直接对比</h3>
    {comparison_table}
    
    <h3>5.2 全部方法 ARI/NMI 结果表</h3>
    <input type="text" class="search-box" id="tableSearch" placeholder="🔍 搜索方法名..." onkeyup="filterTable()">
    <table class="data-table" id="resultTable">
        <thead>
            <tr>
                <th onclick="sortTable(0)">Method ↕</th>
                <th onclick="sortTable(1)">D1 ARI ↕</th><th onclick="sortTable(2)">D1 NMI ↕</th>
                <th onclick="sortTable(3)">D2 ARI ↕</th><th onclick="sortTable(4)">D2 NMI ↕</th>
                <th onclick="sortTable(5)">D3 ARI ↕</th><th onclick="sortTable(6)">D3 NMI ↕</th>
                <th onclick="sortTable(7)">D4 ARI ↕</th><th onclick="sortTable(8)">D4 NMI ↕</th>
                <th onclick="sortTable(9)">D5 ARI ↕</th><th onclick="sortTable(10)">D5 NMI ↕</th>
                <th onclick="sortTable(11)">Mean ARI ↕</th><th onclick="sortTable(12)">Mean NMI ↕</th>
            </tr>
        </thead>
        <tbody>
            {table_html}
        </tbody>
    </table>
    
    <h3>5.3 单模态分解</h3>
    {single_mod}
    
    <h3>5.4 可视化结果</h3>
    
    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab(event, 'v7aCharts')">v7a Charts</button>
        <button class="tab-btn" onclick="switchTab(event, 'spatialTab')">Spatial Plot</button>
        <button class="tab-btn" onclick="switchTab(event, 'umapTab')">UMAP</button>
    </div>
    
    <div id="v7aCharts" class="tab-content active">
        <h4>v6.1 vs v7a 各数据集对比</h4>
        <div class="figure"><img src="./code/figures_v7a/v61_vs_v7a_comparison.png" alt="v61 vs v7a" onclick="openLightbox(this)"><p class="figure-caption">STAMP v6.1 vs v7a per Dataset</p></div>
        <h4>平均 ARI 排名</h4>
        <div class="figure"><img src="./code/figures_v7a/benchmark_mean_rank_v7a.png" alt="Ranking" onclick="openLightbox(this)"><p class="figure-caption">Mean ARI across 5 datasets (all methods)</p></div>
        <h4>单模态分解</h4>
        <div class="figure"><img src="./code/figures_v7a/single_modality_breakdown.png" alt="Breakdown" onclick="openLightbox(this)"><p class="figure-caption">RNA branch (z_r) vs ATAC branch (z_a)</p></div>
    </div>
    
    <div id="spatialTab" class="tab-content">
        <h4>Dataset 1 — ARI = 0.815</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_spatial_domain_d1.png" alt="D1 Spatial" onclick="openLightbox(this)"></div>
        <h4>Dataset 2 — ARI = 0.863</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_spatial_domain_d2.png" alt="D2 Spatial" onclick="openLightbox(this)"></div>
        <h4>Dataset 3 — ARI = 0.915</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_spatial_domain_d3.png" alt="D3 Spatial" onclick="openLightbox(this)"></div>
        <h4>Dataset 4 — ARI = 0.961</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_spatial_domain_d4.png" alt="D4 Spatial" onclick="openLightbox(this)"></div>
        <h4>Dataset 5 — ARI = 0.962</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_spatial_domain_d5.png" alt="D5 Spatial" onclick="openLightbox(this)"></div>
    </div>
    
    <div id="umapTab" class="tab-content">
        <h4>Dataset 1</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_umap_d1.png" alt="D1 UMAP" onclick="openLightbox(this)"></div>
        <h4>Dataset 2</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_umap_d2.png" alt="D2 UMAP" onclick="openLightbox(this)"></div>
        <h4>Dataset 3</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_umap_d3.png" alt="D3 UMAP" onclick="openLightbox(this)"></div>
        <h4>Dataset 4</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_umap_d4.png" alt="D4 UMAP" onclick="openLightbox(this)"></div>
        <h4>Dataset 5</h4>
        <div class="figure"><img src="./code/figures_v7a/stamp_umap_d5.png" alt="D5 UMAP" onclick="openLightbox(this)"></div>
    </div>
</section>

<section id="analysis">
    <h2>6. 分析讨论</h2>
    
    <h3>6.1 v7a 相较于 v6.1 的提升与代价</h3>
    <div class="card">
        <p><span class="badge badge-success">提升</span></p>
        <ul>
            <li>平均 ARI 从 <b>0.894 → 0.903</b>（+0.01），已超越 STAGATE（0.894）</li>
            <li><b>D1 提升最显著</b>（+0.043）：v6.1 最弱的数据集，双图并行有效弥补了信息盲区</li>
            <li><b>D5 也有不错提升</b>（+0.012）</li>
        </ul>
        <p><span class="badge badge-accent">代价</span></p>
        <ul>
            <li><b>D4 略有下降</b>（-0.014）：已接近天花板（0.975），特征图可能引入轻微过平滑</li>
            <li><b>RNA 单分支普遍下降</b>（D1: 0.52→0.35, D3: 0.64→0.41）：表示更"面向融合优化"而非"单模态聚类"</li>
        </ul>
    </div>
    
    <h3>6.2 双图并行的作用机制</h3>
    <div class="card">
        <table class="comparison-table">
            <tr><th>分支</th><th>v6.1 (单图)</th><th>v7a (双图)</th><th>变化</th></tr>
            <tr><td>z_r (RNA)</td><td>0.519 / 0.142 / 0.637 / 0.620 / 0.497</td><td>0.349 / 0.216 / 0.414 / 0.353 / 0.433</td><td>普遍下降</td></tr>
            <tr><td>z_a (ATAC)</td><td>0.748 / 0.870 / 0.884 / 0.962 / 0.936</td><td>0.827 / 0.893 / 0.919 / 0.962 / 0.960</td><td><b>全面提升</b></td></tr>
        </table>
        <p><b>关键洞察</b>：双图并行对 ATAC 模态的收益远大于 RNA。ATAC 的特征空间结构更清晰（peak-基因调控关系），特征 kNN 图能更准确捕获生物学邻居；而 RNA 受 dropout 影响大，特征图可能引入假阳性边。</p>
    </div>
    
    <h3>6.3 设计选择的有效性验证</h3>
    <table class="comparison-table">
        <tr><th>设计选择</th><th>变体</th><th>Dataset 1 ARI</th><th>结论</th></tr>
        <tr><td><b>双图并行 (v7a)</b></td><td>单图 (v6.1)</td><td>0.772 → <b>0.815</b></td><td><b>有效，+0.043</b></td></tr>
        <tr><td>非对称 CMA</td><td>双向对称</td><td>0.76 → 0.74</td><td>有效保护 ATAC</td></tr>
        <tr><td>跨模态重建</td><td>仅自重建</td><td>0.24</td><td>防止坍塌必要条件</td></tr>
        <tr><td>Stop-gradient 对齐</td><td>直接 MSE</td><td>0.76 → 0.68</td><td>防止梯度坍塌</td></tr>
        <tr><td>PCA 维度</td><td>PCA 100/300</td><td>0.75 / 0.37</td><td>PCA 50 最优</td></tr>
    </table>
    
    <h3>6.4 总结与展望</h3>
    <div class="card">
        <p>STAMP v7a 通过<b>双图并行编码器</b>，在保持 v6.1 核心架构不变的前提下，将平均 ARI 从 <b>0.894 提升至 0.903</b>。</p>
        <p><b>未来方向</b>：</p>
        <ol>
            <li><b>自适应特征图邻居数</b>：当前固定 k=15，可根据局部密度自适应调整（如 D4 可能需要更小的 k）</li>
            <li><b>RNA 分支增强</b>：当前 RNA 单分支平均 ARI 仅 0.35，仍是最大瓶颈</li>
            <li><b>保守的门控修复</b>：在 v7a 基础上叠加初始化 confidence≈0.3 的自适应融合门控</li>
        </ol>
    </div>
</section>

<footer style="text-align:center; padding: 3rem 0; color: var(--text-light); font-size: 0.85rem;">
    <p>STAMP v7a Technical Report | Generated 2026-04-14</p>
    <p>Code: /data/lvyongji/Assignment5/code | Baseline backup: /data/lvyongji/Assignment5/baseline_v6.1/</p>
</footer>

</div>

<div class="lightbox" id="lightbox" onclick="closeLightbox()">
    <img src="" alt="Full size" id="lightboxImg">
</div>

<script>
function copyCode(btn) {{
    const block = btn.closest('.code-block');
    const code = block.querySelector('pre').innerText;
    navigator.clipboard.writeText(code).then(() => {{
        const orig = btn.innerText;
        btn.innerText = 'Copied!';
        setTimeout(() => btn.innerText = orig, 1500);
    }});
}}
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
function openLightbox(img) {{
    document.getElementById('lightboxImg').src = img.src;
    document.getElementById('lightbox').classList.add('active');
}}
function closeLightbox() {{
    document.getElementById('lightbox').classList.remove('active');
}}
function switchTab(evt, tabId) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    evt.target.classList.add('active');
}}
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
        if (isNumeric) {{ av = parseFloat(av); bv = parseFloat(bv); }}
        if (dir === 'asc') return av > bv ? 1 : -1;
        return av < bv ? 1 : -1;
    }});
    rows.forEach(r => tbody.appendChild(r));
}}
function filterTable() {{
    const query = document.getElementById('tableSearch').value.toLowerCase();
    document.querySelectorAll('#resultTable tbody tr').forEach(row => {{
        const method = row.cells[0].innerText.toLowerCase();
        row.style.display = method.includes(query) ? '' : 'none';
    }});
}}
const stampRows = document.querySelectorAll('#resultTable tbody tr[data-method="STAMPv7"]');
stampRows.forEach(row => {{
    row.style.background = '#fef2f2';
    row.querySelector('.method-name').innerHTML = '<span class="stamp">STAMPv7 ⭐</span>';
}});
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

print("Updated report generated: /data/lvyongji/Assignment5/report.html")
