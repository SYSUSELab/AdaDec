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

# Data
L = np.array([2, 3, 4, 5, 6, 7, 8, 9])
pass1 = np.array([54.27, 56.10, 56.71, 57.32, 54.27, 55.49, 55.49, 55.49])
baseline = 51.83

# Placeholder symmetric error bars: replace with real errors if you have them
# Here we choose a small value to make error markers visible but not dominate.
y_err = np.full_like(pass1, 0.30)  # <-- replace with actual std errs / CI if available

# Font configuration (prefer Times New Roman; fall back gracefully)
preferred_font = "DejaVu Sans"
available = [f.name for f in font_manager.fontManager.ttflist]
if preferred_font in available:
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = [preferred_font]
else:
    # fallback to a common serif font available in matplotlib (DejaVu Serif)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['DejaVu Serif']

# Ensure PDF uses embedded TrueType fonts for better cross-system rendering in journals
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42

# Text sizes (in points) per user's spec
title_fs = 14
axis_label_fs = 12
tick_label_fs = 10
legend_fs = 9
annotation_fs = 9

# Figure layout
fig_width = 6.5  # inches, typical single-column figure width for journals
fig_height = 4.0
fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

# Plot line with error bars and circular markers
line = ax.errorbar(
    L, pass1, yerr=y_err,
    fmt='-o', linewidth=1.4, markersize=6, markeredgewidth=1.0,
    markeredgecolor='black', markerfacecolor='white', capsize=4, elinewidth=1.0,
    label='DS-1.3B (HumanEval+)'
)

# Baseline line (grey dashed) and legend entry showing exact value
ax.axhline(baseline, linestyle='--', linewidth=1.2, color='gray', zorder=0)
# Add baseline to legend with formatted value
baseline_label = f'Baseline = {baseline:.2f}%'
# Create a custom legend entry for baseline
from matplotlib.lines import Line2D
baseline_handle = Line2D([0], [0], color='gray', linestyle='--', linewidth=1.2)
ax.legend([line[0], baseline_handle], ['DS-1.3B (HumanEval+)', baseline_label],
          fontsize=legend_fs, frameon=False, loc='lower right', prop={'weight': 'medium'})

# Axis labels and title
ax.set_xlabel('Lookahead Length (L)', fontsize=axis_label_fs, fontweight='medium')
ax.set_ylabel('pass@1 (%)', fontsize=axis_label_fs, fontweight='medium')
# ax.set_title('Sensitivity of DS-1.3B pass@1 to Lookahead Length (L)', fontsize=title_fs)

# Ticks and grid
ax.set_xticks(L)
ax.set_xticklabels([str(x) for x in L], fontsize=tick_label_fs, fontweight='medium')
ax.tick_params(axis='y', labelsize=tick_label_fs)
y_ticks = ax.get_yticks()  # 获取当前y轴刻度位置
ax.set_yticks(y_ticks)     # 显式设置y轴刻度位置
ax.set_yticklabels(ax.get_yticks(), fontsize=tick_label_fs, fontweight='medium')
# Horizontal grid only
ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
ax.xaxis.grid(False)

# Y-limits: add small margins
y_min = min(pass1.min(), baseline) - 3.0
y_max = max(pass1.max(), baseline) + 2.0
ax.set_ylim(y_min, y_max)

# Annotate each data point with its numeric value (2 decimal places)
# 找到最大值对应的索引
max_idx = np.argmax(pass1)
max_y = pass1[max_idx]

# 标注每个数据点，只加粗最大值点
for i, (xi, yi) in enumerate(zip(L, pass1)):
    # 判断是否为最大值点
    if i == max_idx:
        # 最大值点使用粗体
        ax.annotate(f'{yi:.2f}%', xy=(xi, yi), xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom', fontsize=annotation_fs, fontweight='bold')
    else:
        # 其他点使用普通字重
        ax.annotate(f'{yi:.2f}%', xy=(xi, yi), xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom', fontsize=annotation_fs, fontweight='medium')

# Emphasize the peak at L=5: larger marker and subtle annotation
peak_idx = np.argmax(pass1)
peak_x = L[peak_idx]
peak_y = pass1[peak_idx]
# Draw a highlighted marker
ax.plot(peak_x, peak_y, marker='o', markersize=9, markeredgewidth=1.4,
        markeredgecolor='black', markerfacecolor='gold', zorder=5)
# ax.annotate('Peak (L=5)', xy=(peak_x+0.15, peak_y), xytext=(10, 10),
#             textcoords='offset points', fontsize=annotation_fs, fontweight='medium',
#             arrowprops=dict(arrowstyle='->', lw=0.8))

# Save outputs (PNG 300 DPI and vector PDF)
output_dir = r'C:\Users\arw\Desktop'
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, 'L_analysis.pdf')
fig.savefig(pdf_path)

# Also show the figure inline (the notebook will display it)
# plt.show()

