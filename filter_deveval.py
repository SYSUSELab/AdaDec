import json
import random

def filter_jsonl_by_prompt_lines(input_file, output_file, max_lines, num_records, 
                                random_seed=42, line_numbers_file=None):
    """
    从 input_file 读取 jsonl，筛选出 prompt 字段行数不超过 max_lines 的记录，
    从中随机选择 num_records 条记录，并将结果写入 output_file。
    同时记录并输出选中记录在原始文件中的行号（从1开始）。
    通过设置 random_seed 确保结果可复现。
    """
    # 收集所有符合条件的记录及其行号
    valid_records = []  # 元素为元组 (行号, 记录数据)
    
    with open(input_file, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):  # 从1开始计数行号
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                data = json.loads(stripped_line)
            except json.JSONDecodeError:
                continue  # 跳过不合法的 json 行
            
            prompt = data.get("prompt", "")
            if isinstance(prompt, str):
                line_count = prompt.count("\n") + 1 if prompt else 0
                if line_count <= max_lines:
                    valid_records.append((line_num, data))
    
    # 设置随机种子，确保结果可复现
    random.seed(random_seed)
    
    # 随机选择指定数量的记录，如果符合条件的记录不足，则选择全部
    selected_count = min(num_records, len(valid_records))
    selected_records = random.sample(valid_records, selected_count)
    
    # 提取选中的行号并排序（便于阅读）
    selected_line_numbers = sorted([line_num for line_num, _ in selected_records])
    
    # 将选中的记录写入输出文件
    with open(output_file, "w", encoding="utf-8") as fout:
        for _, record in selected_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # 输出或保存行号信息
    if line_numbers_file:
        with open(line_numbers_file, "w", encoding="utf-8") as ln_file:
            for line_num in selected_line_numbers:
                ln_file.write(f"{line_num}\n")
    
    return selected_count, len(valid_records), selected_line_numbers

if __name__ == "__main__":
    # 配置参数
    input_path = "deveval_data.jsonl"
    output_path = "new_deveval_filtered_data.jsonl"
    line_numbers_path = "selected_line_numbers.txt"  # 保存选中行号的文件
    max_allowed_lines = 200  # prompt最大允许行数
    num_to_select = 238       # 需要随机选择的记录数量
    random_seed = 42          # 随机种子，用于保证结果可复现
    
    selected, total_valid, line_numbers = filter_jsonl_by_prompt_lines(
        input_path, output_path, max_allowed_lines, num_to_select, 
        random_seed, line_numbers_path
    )
    
    print(f"筛选完成，共从 {total_valid} 条符合条件的记录中随机选择了 {selected} 条")
    print(f"选中的记录已保存到 {output_path}")
    print(f"选中记录的原始行号（从1开始）已保存到 {line_numbers_path}")
    print(f"选中的行号：{line_numbers}")
    