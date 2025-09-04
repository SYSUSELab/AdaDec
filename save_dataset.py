from datasets import load_dataset
import json

def build_prompt(example: dict) -> str:
    """
    从 mbppplus 的一条样本构造 prompt:
    [可能的 import 代码] + def 函数签名 + docstring(来自prompt) + # YOUR CODE HERE
    """
    code = example["code"].strip().splitlines()

    def_index = None
    for i, line in enumerate(code):
        if line.strip().startswith("def "):
            def_index = i
            break
    if def_index is None:
        raise ValueError(f"无法在 code 中找到函数定义: {example['code'][:80]}")

    # 前面的 import / 辅助代码
    prefix_code = "\n".join(code[:def_index])

    # 函数定义行
    def_line = code[def_index]

    # 题目描述放在 docstring
    docstring = example["prompt"].strip()

    # 组装
    prompt_parts = []
    if prefix_code:
        prompt_parts.append(prefix_code)  # import 语句
    prompt_parts.append(f"""{def_line}
    \"\"\" {docstring} \"\"\"
    # YOUR CODE HERE""")

    return "\n".join(prompt_parts) + "\n"


# 加载数据集
ds = load_dataset("evalplus/mbppplus")
problems = ds["test"]

# 保存为 jsonl 文件
output_file = "mbppplus_test.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for problem in problems:
        problem['prompt'] = build_prompt(problem)
        # 将每个 problem 转为 JSON 字符串并写入一行
        f.write(json.dumps(problem, ensure_ascii=False) + "\n")

print(f"Saved {len(problems)} problems to {output_file}")