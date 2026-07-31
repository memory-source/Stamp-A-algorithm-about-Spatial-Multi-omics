"""
Regenerate benchmark plots with new styling requirements.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

with open('/data/lvyongji/Assignment5/code/benchmark_results.json') as f:
    results = json.load(f)

datasets = ['D1','D2','D3','D4','D5']

# STAMP results
stamp_results = {
    '1': {'ARI': 0.8151, 'NMI': 0.7544},
    '2': {'ARI': 0.8633, 'NMI': 0.8015},
    '3': {'ARI': 0.9153, 'NMI': 0.8937},
    '4': {'ARI': 0.9614, 'NMI': 0.9434},
    '5': {'ARI': 0.9619, 'NMI': 0.9431},
}

# === Plot 1: benchmark_mean_rank.png ===
# Boxplot + swarm plot, remove STmultiGRN, highlight STAMP only, grey others
# Remove top/right spines, bold borders/ticks, consistent fonts

methods_all = sorted(set(m for r in results.values() for m in r.keys()))
# Remove STmultiGRN and old STAMP (v6.1)
methods_all = [m for m in methods_all if m not in ['STmultiGRN', 'STAMP']]

# Build data for boxplot: each method has up to 5 ARI values
# Include STAMP results
method_data = {}
for m in methods_all:
    vals = [results[str(i)][m]['ARI'] for i in range(1,6) if m in results[str(i)]]
    method_data[m] = vals
method_data['STAMP'] = [stamp_results[str(i)]['ARI'] for i in range(1,6)]

# Sort by mean ARI descending
mean_aris = {m: np.mean(v) for m, v in method_data.items()}
sorted_methods = sorted(mean_aris.keys(), key=lambda x: mean_aris[x], reverse=True)

# Colors: STAMP red, others grey
colors_map = {}
for m in sorted_methods:
    if m == 'STAMP':
        colors_map[m] = '#e74c3c'
    else:
        colors_map[m] = '#95a5a6'

fig, ax = plt.subplots(figsize=(10, 7))

font_size = 14

data_for_box = [method_data[m] for m in sorted_methods]
colors_box = [colors_map[m] for m in sorted_methods]

bp = ax.boxplot(data_for_box, vert=False, patch_artist=True,
                widths=0.5, showfliers=False,
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2))

for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.2)

# Swarm plot: overlay individual points
for i, (m, vals) in enumerate(zip(sorted_methods, data_for_box)):
    color = colors_map[m]
    y_pos = np.random.normal(i + 1, 0.04, size=len(vals))
    ax.scatter(vals, y_pos, color=color, edgecolor='black', s=60, zorder=3, alpha=0.9)

ax.set_yticks(np.arange(1, len(sorted_methods) + 1))
ax.set_yticklabels(sorted_methods, fontsize=font_size)
ax.set_xlabel('ARI', fontsize=font_size)
ax.set_title('ARI Distribution Across 5 Datasets', fontsize=font_size + 2)
ax.set_xlim(-0.05, 1.05)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params(axis='both', which='major', width=2, length=6, labelsize=font_size)

fig.tight_layout()
fig.savefig('/data/lvyongji/Assignment5/code/figures/benchmark_mean_rank.png', dpi=300)
plt.close()

# Also save to figures_v7a for report use
fig.savefig('/data/lvyongji/Assignment5/code/figures_v7a/benchmark_mean_rank.png', dpi=300)
plt.close()

# Also save clean version to figures/
fig.savefig('/data/lvyongji/Assignment5/code/figures/benchmark_mean_rank.png', dpi=300)
plt.close()

# === Plot 2: benchmark_heatmap.png ===
# Remove MUSE, MISO, STmultiGRN
methods_hm = [m for m in methods_all if m not in ['MUSE', 'MISO']]
methods_hm = [m for m in methods_hm if m != 'STAMP']
methods_hm.append('STAMP')

ari_df = pd.DataFrame(index=methods_hm, columns=datasets)
nmi_df = pd.DataFrame(index=methods_hm, columns=datasets)
for i in range(1,6):
    d = str(i)
    for m in methods_all:
        if m in results[d] and m not in ['MUSE', 'MISO', 'STmultiGRN']:
            ari_df.loc[m, f'D{i}'] = results[d][m]['ARI']
            nmi_df.loc[m, f'D{i}'] = results[d][m]['NMI']
    ari_df.loc['STAMP', f'D{i}'] = stamp_results[d]['ARI']
    nmi_df.loc['STAMP', f'D{i}'] = stamp_results[d]['NMI']

ari_df['Mean'] = ari_df.astype(float).mean(axis=1)
nmi_df['Mean'] = nmi_df.astype(float).mean(axis=1)
ari_df = ari_df.sort_values('Mean', ascending=False)
nmi_df = nmi_df.loc[ari_df.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
# ARI heatmap
ari_plot = ari_df.drop('Mean', axis=1).astype(float)
im1 = axes[0].imshow(ari_plot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
axes[0].set_xticks(np.arange(len(datasets)))
axes[0].set_xticklabels(datasets, fontsize=12)
axes[0].set_yticks(np.arange(len(ari_plot.index)))
axes[0].set_yticklabels(ari_plot.index, fontsize=12)
axes[0].set_title('ARI', fontsize=14)
for i in range(len(ari_plot.index)):
    for j in range(len(datasets)):
        val = ari_plot.iloc[i, j]
        if not np.isnan(val):
            axes[0].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, color='black' if val > 0.5 else 'white')
fig.colorbar(im1, ax=axes[0])

# NMI heatmap
nmi_plot = nmi_df.drop('Mean', axis=1).astype(float)
im2 = axes[1].imshow(nmi_plot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
axes[1].set_xticks(np.arange(len(datasets)))
axes[1].set_xticklabels(datasets, fontsize=12)
axes[1].set_yticks(np.arange(len(nmi_plot.index)))
axes[1].set_yticklabels(nmi_plot.index, fontsize=12)
axes[1].set_title('NMI', fontsize=14)
for i in range(len(nmi_plot.index)):
    for j in range(len(datasets)):
        val = nmi_plot.iloc[i, j]
        if not np.isnan(val):
            axes[1].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, color='black' if val > 0.5 else 'white')
fig.colorbar(im2, ax=axes[1])

fig.tight_layout()
fig.savefig('/data/lvyongji/Assignment5/code/figures/benchmark_heatmap.png', dpi=300)
plt.close()

# Save updated CSVs
ari_df.to_csv('/data/lvyongji/Assignment5/code/benchmark_ari.csv')
nmi_df.to_csv('/data/lvyongji/Assignment5/code/benchmark_nmi.csv')

print("Benchmark plots regenerated.")
