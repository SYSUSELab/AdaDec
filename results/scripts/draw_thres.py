import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_list = {'deepseek-1.3b': 'DS-1.3B', 'deepseek-6.7b': 'DS-6.7B', 
              'stable-3b': 'ST-3B',
              'qwen3-0.6b': 'QW-0.6B', 'qwen3-1.7b': 'QW-1.7B', 'qwen3-4b': 'QW-4B', 'qwen3-8b': 'QW-8B'}

thresholds = np.arange(0.6, 1.6, 0.1)

results = []

for model, MD in model_list.items():
    df = pd.read_parquet(f'data/gt_guide_data/{model}_statistics.parquet')
    df = df[df['Rank'] <= 50]
    
    for threshold in thresholds:
        percentile = (df['Entropy'] > threshold).mean() * 100
        rank_lt = df[df['Entropy'] <= threshold]['Rank'].mean()
        rank_ge = df[df['Entropy'] > threshold]['Rank'].mean()
        
        results.append({
            'model': MD,
            'threshold': threshold,
            'Percentile': percentile,
            'Rank_lt': rank_lt,
            'Rank_ge': rank_ge
        })

results_df = pd.DataFrame(results)



plt.figure(figsize=(10, 9))

cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(model_list))]

for i, (model, MD) in enumerate(model_list.items()):
    model_data = results_df[results_df['model'] == MD]
    plt.plot(model_data['threshold'], model_data['Percentile'], 
             marker='o', color=colors[i], label=MD, linewidth=2)

plt.xlabel('Entropy Threshold', fontsize=22)
plt.ylabel('Percentage of Decoding Steps (Entropy > Threshold)', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=17.5, frameon=False)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('results/outputs/thres_tok_pct.pdf')

plt.figure(figsize=(10, 9))

cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(model_list))]

for i, (model, MD) in enumerate(model_list.items()):
    model_data = results_df[results_df['model'] == MD]
    base_color = colors[i]
    
    plt.plot(model_data['threshold'], model_data['Rank_lt'], 
             marker='o', linestyle='-', color=base_color, linewidth=2,
             label=f'{MD} (≤ Threshold)')
    
    plt.plot(model_data['threshold'], model_data['Rank_ge'], 
             marker='x', linestyle='--', color=base_color, linewidth=2,
             label=f'{MD} (> Threshold)')

plt.xlabel('Entropy Threshold', fontsize=22)
plt.ylabel('Average Rank of Ground-truth Tokens', fontsize=22)
plt.legend(loc='upper left', fontsize=16, ncol=2, frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(top=results_df[['Rank_lt', 'Rank_ge']].max().max() * 1.25)
plt.tight_layout()
plt.savefig('results/outputs/thres_avg_rank.pdf')