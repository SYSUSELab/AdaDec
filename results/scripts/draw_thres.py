import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_list = {'deepseek-1.3b': 'DS-1.3B', 'deepseek-6.7b': 'DS-6.7B', 
              'stable-3b': 'ST-3B', 'codellama-7b': 'CL-7B', 
              'qwen3-0.6b': 'QW-0.6B', 'qwen3-1.7b': 'QW-1.7B', 'qwen3-4b': 'QW-4B', 'qwen3-8b': 'QW-8B'}

# Generate 10 entropy thresholds from 0.6 to 1.5 with step 0.1
thresholds = np.arange(0.6, 1.6, 0.1)

# Initialize a list to store results
results = []

# Process each model's Parquet file
for model, MD in model_list.items():
    # Read the Parquet file
    df = pd.read_parquet(f'data/gt_guide_data/{model}_statistics.parquet')
    df = df[df['Rank'] <= 50]
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Percentile: percentage of Entropy values <= threshold
        percentile = (df['Entropy'] > threshold).mean() * 100
        # Rank_lt: average Rank where Entropy <= threshold
        rank_lt = df[df['Entropy'] <= threshold]['Rank'].mean()
        # Rank_ge: average Rank where Entropy > threshold
        rank_ge = df[df['Entropy'] > threshold]['Rank'].mean()
        
        # Store the results
        results.append({
            'model': MD,
            'threshold': threshold,
            'Percentile': percentile,
            'Rank_lt': rank_lt,
            'Rank_ge': rank_ge
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# the result when threshold = 1.0 
# print("\n=== Entropy Threshold = 1.0 ===")
# for _, row in results_df.iterrows():
#     if 0.99 < row['threshold'] < 1.01:
#         print(f"Model: {row['model']}")
#         print(f"  Percentile (Entropy > 1.0): {row['Percentile']:.2f}%")
#         print(f"  Rank (Entropy <= 1.0): {row['Rank_lt']:.2f}")
#         print(f"  Rank (Entropy > 1.0): {row['Rank_ge']:.2f}")
#         print("-" * 40)

# Plot 1
plt.figure(figsize=(10, 6))
for model, MD in model_list.items():
    model_data = results_df[results_df['model'] == MD]
    plt.plot(model_data['threshold'], model_data['Percentile'], marker='o', label=MD)
plt.xlabel('Entropy Threshold', fontsize=14)
plt.ylabel('Percentage of Decoding Steps (Entropy > Threshold)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('results/outputs/thres_tok_pct.pdf')

from cycler import cycler

num_models = len(model_list)
cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in range(num_models)]

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_prop_cycle(cycler('color', colors))

for i, (model, MD) in enumerate(model_list.items()):
    model_data = results_df[results_df['model'] == MD]
    color = colors[i]
    plt.plot(model_data['threshold'], model_data['Rank_lt'], marker='o', linestyle='-', color=color, label=f'{MD} Rank (<= Threshold)')
    plt.plot(model_data['threshold'], model_data['Rank_ge'], marker='x', linestyle='--', color=color, label=f'{MD} Rank (> Threshold)')

plt.xlabel('Entropy Threshold', fontsize=15)
plt.ylabel('Average Rank of Ground-truth Tokens', fontsize=15)
plt.legend(loc='upper left', fontsize=10.5, ncol=2)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.ylim(top=results_df[['Rank_lt', 'Rank_ge']].max().max() * 1.25)
plt.tight_layout()
plt.savefig('results/outputs/thres_avg_rank.pdf')
