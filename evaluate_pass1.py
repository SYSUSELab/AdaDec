import os
import json

BASE_DIR = "experiments"

MODELS = [
    "deepseek-1.3b", "deepseek-6.7b", 
    "stable-3b", 
    "qwen2.5-1.5b", "qwen2.5-7b", 
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", 
]

METHODS = {
    "Greedy": "greedy.jsonl",
    "Beam": "beam_search.jsonl",
    "AdapT": "adapt.jsonl",
    "AdaDec": "adadec.jsonl"
}

def calculate_pass_at_1(filepath, eval_type):
    if not os.path.exists(filepath):
        return "-"
    
    pass_count = 0
    total_count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                total_count += 1
                
                if eval_type in ['humaneval_plus', 'mbpp_plus', 'threshold', 'L_analysis']:
                    if "base_status" in data and "plus_status" in data:
                        if data["base_status"] == "pass" and data["plus_status"] == "pass":
                            pass_count += 1
                    elif "evaluation" in data:
                        if data["evaluation"] is True or str(data["evaluation"]).lower() == "true":
                            pass_count += 1
                
                elif eval_type == 'deveval':
                    if data.get("Result") == "Pass":
                        pass_count += 1
                        
        if total_count == 0:
            return "-"
        return f"{(pass_count / total_count * 100):.2f}"
    except Exception as e:
        return f"Err"

def print_markdown_table(title, headers, rows):
    print(f"### {title}\n")
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"
    print(header_line)
    print(sep_line)
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print("\n" + "="*50 + "\n")

def main():
    datasets = {
        "HumanEval+": "humaneval_plus",
        "MBPP+": "mbpp_plus",
        "DevEval": "deveval"
    }

    for display_name, dir_name in datasets.items():
        headers = ["Model"] + list(METHODS.keys())
        rows = []
        for model in MODELS:
            row = [model]
            for method_name, filename in METHODS.items():
                filepath = os.path.join(BASE_DIR, dir_name, model, filename)
                val = calculate_pass_at_1(filepath, dir_name)
                row.append(val)
            rows.append(row)
        print_markdown_table(f"Table: {display_name} Pass@1 (%)", headers, rows)

    headers = ["Model", "Threshold 1.2 (Pass@1 %)"]
    rows = []
    for model in MODELS:
        filepath = os.path.join(BASE_DIR, "humaneval_plus", "threshold_1.2", f"{model}.jsonl")
        val = calculate_pass_at_1(filepath, "threshold")
        rows.append([model, str(val)])
    print_markdown_table("Table: HumanEval+ (Fixed Threshold = 1.2)", headers, rows)

    headers = ["L Value", "HumanEval+ Pass@1 (%)"]
    rows = []
    for l_val in range(2, 10):
        filepath = os.path.join(BASE_DIR, "L_analysis", f"humanevalplus-ds1.3b-L{l_val}.jsonl")
        val = calculate_pass_at_1(filepath, "L_analysis")
        rows.append([f"L={l_val}", str(val)])
    print_markdown_table("Table: DeepSeek-1.3b L Analysis on HumanEval+", headers, rows)

if __name__ == "__main__":
    main()