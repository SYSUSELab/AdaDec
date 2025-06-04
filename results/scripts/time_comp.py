import pandas as pd

data = {
    'Model': ['DS-1.3B', 'DS-6.7B', 'ST-3B', 'CL-7B', 'QW-0.6B', 'QW-1.7B', 'QW-4B', 'QW-8B'],
    'HumanEval': {
        'Greedy': [193.14, 404.52, 246.32, 823.42, 601.08, 442.26, 686.39, 828.02],
        'Beam Search': [341.46, 1028.10, 470.81, 2244.48, 499.34, 444.93, 1180.95, 1536.05],
        'AppFiveL': [247.86, 580.60, 319.54, 937.45, 574.56, 547.14, 1200.42, 1149.08],
        'AppDynL': [301.98, 673.14, 357.84, 1051.93, 520.26, 664.78, 1219.42, 1659.85],
    },
    'MBPP': {
        'Greedy': [227.19, 424.24, 233.84, 646.91, 1383.22, 566.22, 542.01, 806.38],
        'Beam Search': [286.19, 594.06, 295.98, 1055.80, 413.62, 513.58, 419.42, 687.51],
        'AppFiveL': [230.69, 456.57, 197.52, 529.79, 952.70, 567.12, 806.59, 969.51],
        'AppDynL': [353.88, 527.67, 270.86, 730.55, 1143.27, 1092.04, 1029.79, 1137.02],
    }
}

num_humaneval = 164
num_mbpp = 200

# avg time
avg_time = {'HumanEval': {}, 'MBPP': {}}
methods = ['Greedy', 'Beam Search', 'AppFiveL', 'AppDynL']

for task, div in zip(['HumanEval', 'MBPP'], [num_humaneval, num_mbpp]):
    for method in methods:
        avg_time[task][method] = [x / div for x in data[task][method]]

# delta
delta_percent = {'HumanEval': {}, 'MBPP': {}}
for task in ['HumanEval', 'MBPP']:
    for method in methods[1:]:
        delta_percent[task][method] = [
            100 * (m - g) / g for m, g in zip(avg_time[task][method], avg_time[task]['Greedy'])
        ]

avg_delta = {
    task: {method: round(sum(values)/len(values), 2) for method, values in delta_percent[task].items()}
    for task in ['HumanEval', 'MBPP']
}

humaneval_df = pd.DataFrame(avg_time['HumanEval'], index=data['Model'])
mbpp_df = pd.DataFrame(avg_time['MBPP'], index=data['Model'])

delta_h_df = pd.DataFrame(delta_percent['HumanEval'], index=data['Model'])
delta_m_df = pd.DataFrame(delta_percent['MBPP'], index=data['Model'])

humaneval_df = humaneval_df.round(2)
mbpp_df = mbpp_df.round(2)
delta_h_df = delta_h_df.round(2)
delta_m_df = delta_m_df.round(2)

print("=== Avg Time (HumanEval) ===")
print(humaneval_df.round(2))
print("\n=== Avg Time (MBPP) ===")
print(mbpp_df.round(2))
print("\n===  Delta (HumanEval) ===")
print(delta_h_df.round(2))
print("\n=== Delta (MBPP) ===")
print(delta_m_df.round(2))
print("\n=== Avg Delta ===")
print(avg_delta)