import json

def filter_jsonl_by_prompt_lines(input_file, output_file, max_lines):
    """
    从 input_file 读取 jsonl，筛选掉 prompt 字段行数超过 max_lines 的记录，
    并将结果写入 output_file。
    """
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
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
                if line_count <= max_lines:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # 修改这里的路径和行数阈值
    input_path = "deveval_data.jsonl"
    output_path = "deveval_filtered_data.jsonl"
    max_allowed_lines = 1000
    
    filter_jsonl_by_prompt_lines(input_path, output_path, max_allowed_lines)
    print(f"筛选完成，结果已保存到 {output_path}")