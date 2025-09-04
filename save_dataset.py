from datasets import load_dataset
import json

# 加载数据集
ds = load_dataset("evalplus/mbppplus")
problems = ds["test"]

# 保存为 jsonl 文件
output_file = "mbppplus_test.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for problem in problems:
        # 将每个 problem 转为 JSON 字符串并写入一行
        f.write(json.dumps(problem, ensure_ascii=False) + "\n")

print(f"Saved {len(problems)} problems to {output_file}")