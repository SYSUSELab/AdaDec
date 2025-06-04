import re
import matplotlib.pyplot as plt
import numpy as np

def extract_failed_ids(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in reversed(lines):
        match = re.search(r"failed task ids: (\[.*\])", line)
        if match:
            failed_ids_str = match.group(1)
            failed_ids = eval(failed_ids_str)  # Note: Trust the file contents
            return set(failed_ids)
    return set()

def extract_entropy_values(file_path):
    failed_ids = extract_failed_ids(file_path)
    all_entropy = []
    target_entropy = []

    current_task = None
    i = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    while i < len(lines):
        line = lines[i]

        task_match = re.search(r'##### (HumanEval/\d+) ######', line)
        if task_match:
            current_task = task_match.group(1)
            i += 1
            continue

        if current_task not in failed_ids:
            i += 1
            continue

        info_match = re.match(r'\[INFO\].*STEP:\d+(.*?)$', line)
        if info_match:
            has_target = 'TARGET' in info_match.group(1)

            j = i + 1
            while j < len(lines):
                entropy_match = re.search(r'entropy:tensor\(\[([0-9.]+)\]', lines[j])
                if entropy_match:
                    entropy_val = float(entropy_match.group(1))
                    all_entropy.append(entropy_val)
                    if has_target:
                        target_entropy.append(entropy_val)
                    break
                j += 1
            i = j
        else:
            i += 1

    # Rest entropy = All - Target
    rest_entropy = [val for val in all_entropy if val not in target_entropy]
    return rest_entropy, target_entropy

def print_boxplot_stats(data, label):
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    min_val = np.min(data)
    max_val = np.max(data)
    mean = np.mean(data)
    outliers = [x for x in data if x < lower_bound or x > upper_bound]

    print(f"\n[{label}]")
    print(f"Number of samples: {len(data)}")
    print(f"Q1 (25%): {q1:.4f}")
    print(f"Median (Q2): {q2:.4f}")
    print(f"Q3 (75%): {q3:.4f}")
    print(f"Mean: {mean:.4f}")
    print(f"IQR: {iqr:.4f}")

def draw_boxplots(rest_entropy, target_entropy):
    print_boxplot_stats(rest_entropy, "Rest Entropy")
    print_boxplot_stats(target_entropy, "Target Entropy")

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [rest_entropy, target_entropy],
        labels=['Non-drift Steps', 'Drift Points'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue'),
        medianprops=dict(color='red')
    )
    plt.ylabel('Entropy Distribution', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/outputs/entropy_distribution.pdf')

log_file_path = 'data/deepseek-6.7b_sample.log'
rest_entropy, target_entropy = extract_entropy_values(log_file_path)
draw_boxplots(rest_entropy, target_entropy)
