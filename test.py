import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def find_empty_completions(jsonl_file_path):
    """
    从jsonl文件中读取每一行的json对象的completion字段，
    并输出completion字段为空字符串的行号（从1开始计数）。

    参数:
    jsonl_file_path (str): jsonl文件的路径

    返回:
    list: 包含所有completion字段为空的行号的列表
    """
    empty_completion_lines = []
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    json_obj = json.loads(line)
                    if 'completion' in json_obj and json_obj['completion'] == "":
                        empty_completion_lines.append(line_num)
                except json.JSONDecodeError as e:
                    print(f"警告：第 {line_num} 行不是有效的JSON格式，已跳过。错误信息: {e}")
    except FileNotFoundError:
        print(f"错误：文件 {jsonl_file_path} 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    return empty_completion_lines

def analyze_selected_lines_char_distribution(jsonl_file_path: str, line_numbers: List[int]) -> Dict:
    """
    分析jsonl文件中指定行的字符数分布情况
    
    参数:
    jsonl_file_path: jsonl文件路径
    line_numbers: 要分析的行号列表（从1开始）
    
    返回:
    包含分析结果的字典
    """
    
    def get_all_lines_chars(file_path: str) -> List[int]:
        """获取所有行的字符数"""
        all_chars = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_chars.append(len(line.strip()))
        return all_chars
    
    def get_selected_lines_chars(file_path: str, target_lines: List[int]) -> Tuple[List[int], Dict[int, str]]:
        """获取指定行的字符数和内容"""
        selected_chars = []
        selected_contents = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for current_line_num, line in enumerate(f, 1):
                if current_line_num in target_lines:
                    stripped_line = line.strip()
                    selected_chars.append(len(stripped_line))
                    selected_contents[current_line_num] = stripped_line
        
        return selected_chars, selected_contents
    
    # 获取所有行和指定行的字符数
    all_line_chars = get_all_lines_chars(jsonl_file_path)
    selected_line_chars, selected_contents = get_selected_lines_chars(jsonl_file_path, line_numbers)
    
    if not selected_line_chars:
        return {"error": "指定的行号未找到或文件为空"}
    
    # 计算分布情况
    all_array = np.array(all_line_chars)
    selected_array = np.array(selected_line_chars)
    
    # 创建分布统计
    distribution = {
        "file_statistics": {
            "total_lines": len(all_array),
            "mean_chars": float(np.mean(all_array)),
            "median_chars": float(np.median(all_array)),
            "std_chars": float(np.std(all_array)),
            "min_chars": int(np.min(all_array)),
            "max_chars": int(np.max(all_array))
        },
        "selected_lines": {
            "line_numbers": line_numbers,
            "char_counts": selected_line_chars,
            "contents": selected_contents,
            "mean_chars": float(np.mean(selected_array)),
            "median_chars": float(np.median(selected_array)),
            "std_chars": float(np.std(selected_array))
        },
        "distribution_comparison": {
            "percentile_in_file": [float(np.percentile(all_array, p)) for p in [0, 25, 50, 75, 100]],
            "selected_percentiles": [np.searchsorted(np.percentile(all_array, [25, 50, 75]), char_count) for char_count in selected_line_chars],
            "z_scores": (selected_array - np.mean(all_array)) / np.std(all_array)
        }
    }
    
    return distribution

def print_analysis_results(results: Dict):
    """格式化打印分析结果"""
    if "error" in results:
        print(f"错误: {results['error']}")
        return
    
    print("=" * 60)
    print("JSONL文件字符数分布分析报告")
    print("=" * 60)
    
    # 文件整体统计
    file_stats = results["file_statistics"]
    print(f"\n📄 文件整体统计 (总行数: {file_stats['total_lines']}):")
    print(f"   • 平均字符数: {file_stats['mean_chars']:.2f}")
    print(f"   • 中位数: {file_stats['median_chars']:.2f}")
    print(f"   • 标准差: {file_stats['std_chars']:.2f}")
    print(f"   • 范围: {file_stats['min_chars']} - {file_stats['max_chars']}")
    
    # 选定行统计
    selected = results["selected_lines"]
    print(f"\n🔍 选定行分析 (共 {len(selected['line_numbers'])} 行):")
    for line_num, char_count, z_score in zip(selected["line_numbers"], 
                                           selected["char_counts"], 
                                           results["distribution_comparison"]["z_scores"]):
        content = selected["contents"][line_num]
        # 限制显示长度
        preview = (content[:50] + '...') if len(content) > 50 else content
        print(f"   行 {line_num:3d}: {char_count:3d} 字符 | z-score: {z_score:6.2f} | 内容: {preview}")
    
    # 分布比较
    print(f"\n📊 分布比较:")
    percentiles = results["distribution_comparison"]["percentile_in_file"]
    print(f"   文件四分位数: [0%:{percentiles[0]}, 25%:{percentiles[1]}, 50%:{percentiles[2]}, 75%:{percentiles[3]}, 100%:{percentiles[4]}]")
    
    selected_percentiles = results["distribution_comparison"]["selected_percentiles"]
    quartile_names = ["Q1以下", "Q1-Q2", "Q2-Q3", "Q3以上"]
    for line_num, q_idx in zip(selected["line_numbers"], selected_percentiles):
        print(f"   行 {line_num} 位于: {quartile_names[q_idx]}")

def plot_distribution_comparison(results: Dict, save_path: str = None):
    """绘制分布对比图"""
    if "error" in results:
        print(f"无法绘图: {results['error']}")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 整体分布直方图
    all_chars = [results["file_statistics"]["mean_chars"]] * 3  # 占位
    selected_chars = results["selected_lines"]["char_counts"]
    
    # 重新计算实际值
    file_stats = results["file_statistics"]
    ax1.hist(all_chars, bins=50, alpha=0.7, label='all lines', color='skyblue')
    ax1.axvline(file_stats["mean_chars"], color='blue', linestyle='--', 
                label=f'all avg: {file_stats["mean_chars"]:.1f}')
    ax1.set_title('all lines char nums distribution')
    ax1.set_xlabel('char nums')
    ax1.set_ylabel('pin ci')
    ax1.legend()
    
    # 2. 选定行分布
    ax2.hist(selected_chars, bins=20, alpha=0.7, color='lightcoral', label='target lines')
    ax2.axvline(np.mean(selected_chars), color='red', linestyle='--',
                label=f'target line avg: {np.mean(selected_chars):.1f}')
    ax2.set_title('target line char nums distribution')
    ax2.set_xlabel('char nums')
    ax2.set_ylabel('pin ci')
    ax2.legend()
    
    # 3. 箱线图对比
    box_data = [all_chars, selected_chars]
    ax3.boxplot(box_data, labels=['all line', 'target line'])
    ax3.set_title('char nums cmp')
    ax3.set_ylabel('char nums')
    
    # 4. z-score分布
    z_scores = results["distribution_comparison"]["z_scores"]
    ax4.bar(range(len(z_scores)), z_scores, color='purple', alpha=0.7)
    ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(1, color='red', linestyle='--', alpha=0.5, label='+1σ')
    ax4.axhline(-1, color='red', linestyle='--', alpha=0.5, label='-1σ')
    ax4.set_title('选定行的Z-Score分布')
    ax4.set_xlabel('行号索引')
    ax4.set_ylabel('Z-Score')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()

def plot_char_distribution_scatter(jsonl_file_path: str, line_numbers: List[int], 
                                 figsize: Tuple[int, int] = (15, 8), 
                                 point_size: int = 50,
                                 alpha: float = 0.6,
                                 title: str = "JSONL文件字符数分布散点图",
                                 save_path: str = None):
    """
    用散点图绘制所有行的字符数分布，目标行用红色突出显示
    
    参数:
    jsonl_file_path: jsonl文件路径
    line_numbers: 目标行号列表（从1开始）
    figsize: 图表尺寸
    point_size: 点的大小
    alpha: 透明度
    title: 图表标题
    save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # 读取数据并计算字符数
    all_chars = []
    target_chars = []
    target_line_nums = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            char_count = len(line.strip())
            all_chars.append(char_count)
            
            if line_num in line_numbers:
                target_chars.append(char_count)
                target_line_nums.append(line_num)
    
    if not all_chars:
        print("错误: 文件为空或无法读取")
        return
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制所有行（蓝色）
    all_line_nums = list(range(1, len(all_chars) + 1))
    plt.scatter(all_line_nums, all_chars, 
               c='blue', alpha=alpha, s=point_size, 
               label='所有行', zorder=1)
    
    # 高亮目标行（红色）
    if target_chars:
        plt.scatter(target_line_nums, target_chars, 
                   c='red', alpha=alpha*1.2, s=point_size*1.5, 
                   label='目标行', zorder=2, edgecolors='darkred', linewidth=1)
    
    # 添加统计线
    mean_all = sum(all_chars) / len(all_chars)
    plt.axhline(y=mean_all, color='blue', linestyle='--', alpha=0.8, 
                label=f'平均值 ({mean_all:.1f})')
    
    if target_chars:
        mean_target = sum(target_chars) / len(target_chars)
        plt.axhline(y=mean_target, color='red', linestyle='--', alpha=0.8, 
                    label=f'目标行平均值 ({mean_target:.1f})')
    
    # 图表装饰
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('行号', fontsize=12)
    plt.ylabel('字符数', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 创建图例
    blue_patch = mpatches.Patch(color='blue', alpha=alpha, label='所有行')
    red_patch = mpatches.Patch(color='red', alpha=alpha*1.2, label='目标行')
    plt.legend(handles=[blue_patch, red_patch], loc='upper right')
    
    # 优化坐标轴
    plt.xlim(0, len(all_chars) + 1)
    
    # 添加分布背景色
    q25 = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.25
    q50 = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.5
    q75 = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.75
    
    plt.axhspan(plt.ylim()[0], q25, facecolor='gray', alpha=0.05, label='Q1')
    plt.axhspan(q25, q50, facecolor='gray', alpha=0.1, label='Q2')
    plt.axhspan(q50, q75, facecolor='gray', alpha=0.05, label='Q3')
    plt.axhspan(q75, plt.ylim()[1], facecolor='gray', alpha=0.1, label='Q4')
    
    # 添加四分位标注
    plt.text(len(all_chars) + 0.5, q25/2, 'Q1区域', alpha=0.7, fontsize=9, 
             ha='left', va='center', rotation=90)
    plt.text(len(all_chars) + 0.5, (q25+q50)/2, 'Q2区域', alpha=0.7, fontsize=9, 
             ha='left', va='center', rotation=90)
    plt.text(len(all_chars) + 0.5, (q50+q75)/2, 'Q3区域', alpha=0.7, fontsize=9, 
             ha='left', va='center', rotation=90)
    plt.text(len(all_chars) + 0.5, (q75+plt.ylim()[1])/2, 'Q4区域', alpha=0.7, fontsize=9, 
             ha='left', va='center', rotation=90)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"图表已保存至: {save_path}")
        plt.close()
    else:
        plt.show()



# 使用示例
if __name__ == "__main__":
    # 示例用法
    jsonl_path = 'deveval_filtered_data.jsonl'
    target_lines = find_empty_completions('AdapT/deveval_outputs/stabilityai-stable-code-instruct-3b.jsonl')
    
    # 执行分析
    analysis_results = analyze_selected_lines_char_distribution(jsonl_path, target_lines)
    
    # 打印结果
    print_analysis_results(analysis_results)
    
    # 绘制图表（可选）
    plot_distribution_comparison(analysis_results, "char_distribution.png")
    
    # 生成散点图
    plot_char_distribution_scatter(
        jsonl_path, 
        target_lines,
        figsize=(16, 9),      # 适合大屏幕的尺寸
        point_size=60,        # 更大的点
        alpha=0.7,            # 透明度
        title="JSONL文件字符数分布 - 目标行高亮对比",
        save_path="char_dist_scatter.png"  # 取消注释以保存图片
    )