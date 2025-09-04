import json

def prompt_line_stats(input_file, max_lines):
    """
    统计 prompt 字段行数超过 max_lines 的占比。
    """
    total = 0
    exceed = 0

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # 跳过不合法的 json 行

            prompt = data.get("prompt", "")
            if isinstance(prompt, str):
                line_count = prompt.count("\n") + 1 if prompt else 0
                total += 1
                if line_count > max_lines:
                    exceed += 1

    if total == 0:
        return 0.0
    return exceed / total * 100  # 返回百分比

if __name__ == "__main__":
    input_path = "deveval_data.jsonl"
    max_allowed_lines = 1000

    percent = prompt_line_stats(input_path, max_allowed_lines)
    print(f"prompt 行数超过 {max_allowed_lines} 的占比: {percent:.2f}%")
