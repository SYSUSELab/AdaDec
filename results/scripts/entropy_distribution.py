import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# 设置matplotlib的字体和样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

def extract_failed_ids(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in reversed(lines):
        match = re.search(r"failed task ids: (\[.*\])", line)
        if match:
            failed_ids_str = match.group(1)
            failed_ids = eval(failed_ids_str)  # Note: Trust the file contents
            return set(failed_ids)
    return set()

def extract_entropy_values(file_path):
    failed_ids = extract_failed_ids(file_path)
    all_entropy = []
    target_entropy = []

    current_task = None
    i = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    while i < len(lines):
        line = lines[i]

        task_match = re.search(r'##### (HumanEval/\d+) ######', line)
        if task_match:
            current_task = task_match.group(1)
            i += 1
            continue

        if current_task not in failed_ids:
            i += 1
            continue

        info_match = re.match(r'\[INFO\].*STEP:\d+(.*?)$', line)
        if info_match:
            has_target = 'TARGET' in info_match.group(1)

            j = i + 1
            while j < len(lines):
                entropy_match = re.search(r'entropy:tensor\(\[([0-9.]+)\]', lines[j])
                if entropy_match:
                    entropy_val = float(entropy_match.group(1))
                    all_entropy.append(entropy_val)
                    if has_target:
                        target_entropy.append(entropy_val)
                    break
                j += 1
            i = j
        else:
            i += 1

    # Rest entropy = All - Target
    rest_entropy = [val for val in all_entropy if val not in target_entropy]
    return rest_entropy, target_entropy

def print_boxplot_stats(data, label):
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    min_val = np.min(data)
    max_val = np.max(data)
    mean = np.mean(data)
    outliers = [x for x in data if x < lower_bound or x > upper_bound]

    print(f"\n[{label}]")
    print(f"Number of samples: {len(data)}")
    print(f"Q1 (25%): {q1:.4f}")
    print(f"Median (Q2): {q2:.4f}")
    print(f"Q3 (75%): {q3:.4f}")
    print(f"Mean: {mean:.4f}")
    print(f"IQR: {iqr:.4f}")

def draw_boxplots(rest_entropy, target_entropy):
    print_boxplot_stats(rest_entropy, "Rest Entropy")
    print_boxplot_stats(target_entropy, "Target Entropy")

    # 设置现代化的配色方案
    colors = ['#3498db', '#e74c3c']  # 蓝色和红色
    edge_colors = ['#2980b9', '#c0392b']  # 深蓝和深红

    # 创建更大的图形
    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制箱线图
    box_plot = ax.boxplot(
        [rest_entropy, target_entropy],
        labels=['Non-drift Steps', 'Drift Points'],
        patch_artist=True,
        widths=0.6,  # 调整箱体宽度
        medianprops=dict(color='white', linewidth=2.5),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.6)
    )

    # 自定义箱体颜色
    for patch, color, edge_color in zip(box_plot['boxes'], colors, edge_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(edge_color)
        patch.set_alpha(0.8)

    # 添加均值标记
    means = [np.mean(rest_entropy), np.mean(target_entropy)]
    for i, mean_val in enumerate(means, 1):
        ax.scatter(i, mean_val, marker='D', s=60, color='white', 
                edgecolors='black', linewidth=1.5, zorder=10, label='Mean' if i == 1 else "")

    # 美化坐标轴 - 加粗字体
    ax.set_ylabel('Entropy Distribution', fontsize=18, color='#2c3e50', fontweight='bold')
    ax.tick_params(axis='x', labelsize=16, colors='#2c3e50', width=1.5)
    ax.tick_params(axis='y', labelsize=14, colors='#2c3e50', width=1.5)

    # 加粗刻度标签
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # 设置背景颜色
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')

    # 设置更紧凑的y轴范围
    data_min = min(min(rest_entropy), min(target_entropy))
    data_max = max(max(rest_entropy), max(target_entropy))
    margin = (data_max - data_min) * 0.1  # 10%的边距
    ax.set_ylim(data_min - margin, data_max + margin)

    # 添加图例 - 加粗字体且无背景
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors[0], edgecolor=edge_colors[0], alpha=0.8, label='Non-drift Steps'),
        plt.Rectangle((0,0),1,1, facecolor=colors[1], edgecolor=edge_colors[1], alpha=0.8, label='Drift Points'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='white', 
                markeredgecolor='black', markersize=8, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False,  # 去掉边框
            shadow=False,  # 去掉阴影
            fontsize=12, 
            prop={'weight': 'bold'})  # 加粗图例文字

    # 调整边距
    plt.tight_layout()
    
    # 添加统计信息文本框
#     stats_text = f"""Statistical Summary:
# Non-drift: μ={np.mean(rest_entropy):.3f}, σ={np.std(rest_entropy):.3f}
# Drift Points: μ={np.mean(target_entropy):.3f}, σ={np.std(target_entropy):.3f}"""
    
#     ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
#             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
#             facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # 保存图片
    plt.savefig('results/outputs/entropy_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数调用
log_file_path = 'data/deepseek-6.7b_sample.log'
rest_entropy, target_entropy = extract_entropy_values(log_file_path)
draw_boxplots(rest_entropy, target_entropy)