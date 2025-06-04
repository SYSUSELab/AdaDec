import re

# Note: The prerequisite for using this file is to use the logging_detail mode to record detailed information when generating.

model_name_list = ['deepseek-1.3b', 'deepseek-6.7b', 'stable-3b', 'codellama-7b', 'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b']
dataset = 'humaneval'       # humaneval, mbpp
decoding_mode = 'adaFixL'   # greedy, beamsearch, adaFixL, adaDynL

def sum_step_maxima_and_avg_time(log_path):
    step_pattern = re.compile(r'STEP:(\d+)')
    time_pattern = re.compile(r'Total time taken:\s+([\d.]+)\s+seconds')

    last_step = -1
    current_max = 0
    maxima = []
    total_time = None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        step_match = step_pattern.search(line)
        if step_match:
            step = int(step_match.group(1))
            if step < last_step:
                maxima.append(current_max)
                current_max = step
            else:
                current_max = max(current_max, step)
            last_step = step

        time_match = time_pattern.search(line)
        if time_match:
            total_time = float(time_match.group(1))

    if current_max != 0:
        maxima.append(current_max)

    total_max_sum = sum(maxima)

    if total_time:
        avg_per_token = total_time / total_max_sum
        return total_max_sum, total_time, avg_per_token
    else:
        return total_max_sum, None, None

def extract_lookahead_ratio(log_path):
    tradition_times_pattern = re.compile(r'Total tradition_times:(\d+)')
    lookahead_times_pattern = re.compile(r'Total lookahead_times:(\d+)')

    last_tradition = None
    last_lookahead = None

    with open(log_path, 'r') as f:
        for line in f:
            match_tradition = tradition_times_pattern.search(line)
            if match_tradition:
                last_tradition = int(match_tradition.group(1))

            match_lookahead = lookahead_times_pattern.search(line)
            if match_lookahead:
                last_lookahead = int(match_lookahead.group(1))

    if last_tradition is not None and last_lookahead is not None:
        ratio = last_lookahead / (last_lookahead + last_tradition)
        return ratio
    else:
        return None

table_data = []

for model_name in model_name_list:
    filename = rf'experiments/{dataset}_outputs/{model_name}/{decoding_mode}/{model_name}.log'
    total_max_sum, total_time, avg_per_token = sum_step_maxima_and_avg_time(filename)
    lookahead_ratio = extract_lookahead_ratio(filename)

    table_data.append([
        model_name,
        total_max_sum,
        f"{total_time:.2f}" if total_time else "N/A",
        f"{avg_per_token:.4f}" if avg_per_token else "N/A",
        f"{lookahead_ratio:.4f}" if lookahead_ratio is not None else "N/A"
    ])

headers = ["Model", "SumTok", "Time", "AvgTok", "LookaheadRatio"]
col_widths = [max(len(str(row[i])) for row in [headers] + table_data) for i in range(len(headers))]

def format_row(row):
    return "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

print(format_row(headers))
print("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
for row in table_data:
    print(format_row(row))
