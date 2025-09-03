#!/usr/bin/env python3
"""
line_length_dist.py
统计文本文件每行的长度并绘制分布图。
用法：
    python line_length_dist.py your_file.txt
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def read_lengths(file_path: Path):
    """读取文件并返回 (lengths, max_line_no, max_len)。"""
    lengths = []
    max_len = 0
    max_line_no = None
    with file_path.open("rt", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            llen = len(line.rstrip("\n\r"))
            lengths.append(llen)
            if llen > max_len:
                max_len = llen
                max_line_no = idx
    return lengths, max_line_no, max_len

def plot_distribution(lengths, max_len, max_line_no, output=None):
    """绘制长度分布直方图，并标出最长行。"""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins="auto", kde=True, color="#4c72b0")
    plt.axvline(max_len, color='red', lw=2, ls='--',
                label=f'max_len：{max_len} (num: {max_line_no})')
    plt.title("各行长度分布")
    plt.xlabel("length")
    plt.ylabel("num")
    plt.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=300)
        print(f"图表已保存到 {output}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="统计文本文件各行长度并绘制分布图")
    parser.add_argument("file", help="要分析的文本文件")
    parser.add_argument("-o", "--output", help="将图表保存为图片文件（如 dist.png）")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"错误：文件 {file_path} 不存在", file=sys.stderr)
        sys.exit(1)

    lengths, max_line_no, max_len = read_lengths(file_path)
    if not lengths:
        print("文件为空，无数据可绘制。", file=sys.stderr)
        sys.exit(0)

    print(f"总行数：{len(lengths)}")
    print(f"最长行：第 {max_line_no} 行，长度 {max_len}")
    plot_distribution(lengths, max_len, max_line_no, args.output)

if __name__ == "__main__":
    main()