import json
import re

def process_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            prompt = data["prompt"]

            # 提取原始代码块
            match = re.search(r"The code to be completed is:\s*```Python(.*?)```", prompt, re.S)
            if match:
                code_block = match.group(1).strip("\n")

                lines = code_block.splitlines()
                normalized_lines = []
                for i, l in enumerate(lines):
                    if i == 0:
                        # def 顶格
                        normalized_lines.append(l.lstrip())
                    else:
                        # 其余行保证至少有 4 空格
                        line = l.lstrip()
                        if line and not line.startswith(" " * 4):
                            line = "    " + line
                        normalized_lines.append(line)

                # 在最后追加 "# YOUR CODE HERE"
                normalized_code = "\n".join(normalized_lines) + "\n    # YOUR CODE HERE"

                # 替换块（去掉末尾三反引号）
                new_block = "Please complete this code: \n```Python\n" + normalized_code + "\n"

                # 替换掉原来的 The code to be completed is 段落
                prompt = re.sub(
                    r"The code to be completed is:\s*```Python.*?```\s*Completed code:",
                    lambda m: new_block,
                    prompt,
                    flags=re.S,
                )

                data["prompt"] = prompt

            # 写回文件
            # f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            print(prompt)
            break


# 使用示例
if __name__ == "__main__":
    process_jsonl("gpt-4-1106_prompt.jsonl", "output.jsonl")
