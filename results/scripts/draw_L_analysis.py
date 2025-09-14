# Python code to generate publication-quality sensitivity analysis plot.
# This will run in the notebook environment and save two files:
# - /mnt/data/DS1.3B_L_sensitivity.png  (300 DPI)
# - /mnt/data/DS1.3B_L_sensitivity.pdf  (vector PDF)
#
# NOTE: The user did not provide per-point error values. For a publication-ready figure,
# replace `y_err` with real standard errors or confidence intervals if available.
# The code uses Times New Roman if available, and falls back to a serif font otherwise.
# The plot emphasizes L=5 peak and shows the baseline as a grey dashed line.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
import os

# data
L = np.array([2, 3, 4, 5, 6, 7, 8, 9])
pass1 = np.array([54.27, 56.10, 56.71, 57.32, 54.27, 55.49, 55.49, 55.49])
baseline = 51.83

y_err = np.full_like(pass1, 0.30)

preferred_font = "DejaVu Sans"
available = [f.name for f in font_manager.fontManager.ttflist]
if preferred_font in available:
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = [preferred_font]
else:
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['DejaVu Serif']

title_fs = 14
axis_label_fs = 12
tick_label_fs = 10
legend_fs = 9
annotation_fs = 9

fig_width = 6.5
fig_height = 4.0
fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

line = ax.errorbar(
    L, pass1, yerr=y_err,
    fmt='-o', linewidth=1.4, markersize=6, markeredgewidth=1.0,
    markeredgecolor='black', markerfacecolor='white', capsize=4, elinewidth=1.0,
    label='DS-1.3B (HumanEval+)'
)

ax.axhline(baseline, linestyle='--', linewidth=1.2, color='gray', zorder=0)
baseline_label = f'Baseline = {baseline:.2f}%'
from matplotlib.lines import Line2D
baseline_handle = Line2D([0], [0], color='gray', linestyle='--', linewidth=1.2)
ax.legend([line[0], baseline_handle], ['DS-1.3B (HumanEval+)', baseline_label],
          fontsize=legend_fs, frameon=False, loc='lower right', prop={'weight': 'medium'})

# Axis labels and title
ax.set_xlabel('Lookahead Length (L)', fontsize=axis_label_fs, fontweight='medium')
ax.set_ylabel('pass@1 (%)', fontsize=axis_label_fs, fontweight='medium')
# ax.set_title('Sensitivity of DS-1.3B pass@1 to Lookahead Length (L)', fontsize=title_fs)

ax.set_xticks(L)
ax.set_xticklabels([str(x) for x in L], fontsize=tick_label_fs, fontweight='medium')
ax.tick_params(axis='y', labelsize=tick_label_fs)
y_ticks = ax.get_yticks()
ax.set_yticks(y_ticks)
ax.set_yticklabels(ax.get_yticks(), fontsize=tick_label_fs, fontweight='medium')
ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
ax.xaxis.grid(False)

y_min = min(pass1.min(), baseline) - 3.0
y_max = max(pass1.max(), baseline) + 2.0
ax.set_ylim(y_min, y_max)

max_idx = np.argmax(pass1)
max_y = pass1[max_idx]

for i, (xi, yi) in enumerate(zip(L, pass1)):
    if i == max_idx:
        ax.annotate(f'{yi:.2f}%', xy=(xi, yi), xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom', fontsize=annotation_fs, fontweight='bold')
    else:
        ax.annotate(f'{yi:.2f}%', xy=(xi, yi), xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom', fontsize=annotation_fs, fontweight='medium')

peak_idx = np.argmax(pass1)
peak_x = L[peak_idx]
peak_y = pass1[peak_idx]
ax.plot(peak_x, peak_y, marker='o', markersize=9, markeredgewidth=1.4,
        markeredgecolor='black', markerfacecolor='gold', zorder=5)


output_dir = 'results/outputs'
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, 'L_analysis.pdf')
fig.savefig(pdf_path)