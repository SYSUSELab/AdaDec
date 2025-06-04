import pandas as pd
from prettytable import PrettyTable

data = {
    "model": ["ds1.3b", "ds6.7b", "stb3b", "clm7b", "qw3-0.6b", "qw3-1.7b", "qw3-4b", "qw3-8b"],
    "human_eval": {
        "greedy":           [93, 118, 91, 63, 35, 76, 94, 101],
        "beam":             [98, 122, 91, 70, 46, 76, 108, 109],
        "ada+fix":        [105,122, 91, 66, 41, 82, 105, 117],
        "ada+dyn":    [101, 116, 91, 70, 42, 78, 110, 115],
        "total": 164
    },
    "mbpp": {
        "greedy":           [73, 95, 70, 70, 24, 68, 47, 81],
        "beam":             [81,102, 75, 73, 41, 80, 70, 98],
        "ada+fix":        [74,102, 80, 77, 36, 74, 78, 98],
        "ada+dyn":    [74, 105, 80, 80, 38, 74, 74, 101],
        "total": 200
    }
}

def percent(v, total):
    return v / total * 100

table = PrettyTable()
table.field_names = ["Model", "HumanEval Greedy", "HumanEval Beam Search", "HumanEval AdaFixL", "HumanEval AdaDynL",
                     "MBPP Greedy", "MBPP Beam Search", "MBPP AdaFixL", "MBPP AdaDynL"]

delta_accumulator = {
    "human_eval": {"beam": [], "ada+fix": [], "ada+dyn": []},
    "mbpp": {"beam": [], "ada+fix": [], "ada+dyn": []}
}

for i, model in enumerate(data["model"]):
    row = [model]
    for task in ["human_eval", "mbpp"]:
        total = data[task]["total"]
        greedy_val = data[task]["greedy"][i]
        greedy_pct = percent(greedy_val, total)
        row.append(f"{greedy_pct:.2f}%")
        for method in ["beam", "ada+fix", "ada+dyn"]:
            val = data[task][method][i]
            pct = percent(val, total)
            delta = pct - greedy_pct
            delta_accumulator[task][method].append(delta)
            row.append(f"{pct:.2f}% (+{delta:.2f}%)")
    table.add_row(row)

avg_row = ["Avg Delta"]
for task in ["human_eval", "mbpp"]:
    total = data[task]["total"]
    greedy_avg = percent(sum(data[task]["greedy"]) / len(data["model"]), total)
    avg_row.append(f"{greedy_avg:.2f}%")
    for method in ["beam", "ada+fix", "ada+dyn"]:
        avg_val = percent(sum(data[task][method]) / len(data["model"]), total)
        avg_delta = sum(delta_accumulator[task][method]) / len(delta_accumulator[task][method])
        avg_row.append(f"{avg_val:.2f}% (+{avg_delta:.2f}%)")

table.add_row(avg_row)

print(table)