import pandas as pd
from scipy.stats import spearmanr

model_name_list = [
    'deepseek-1.3b', 'deepseek-6.7b', 'stable-3b', 'codellama-7b',
    'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b'
]

results = []

for model_name in model_name_list:
    file_path = f'data/gt_guide_data/{model_name}_statistics.parquet'
    try:
        df = pd.read_parquet(file_path)
        sampleNum = len(df)
        if 'Rank' in df.columns and 'Entropy' in df.columns:
            correlation, p_value = spearmanr(df['Rank'], df['Entropy'])
            results.append({
                'Model': model_name,
                'Spearman Correlation': correlation,
                'p-value': p_value,
                'sampleNum': sampleNum
            })
        else:
            results.append({
                'Model': model_name,
                'Spearman Correlation': 'Missing columns',
                'p-value': '',
                'sampleNum': sampleNum
            })
    except Exception as e:
        results.append({
            'Model': model_name,
            'Spearman Correlation': f'Error: {e}',
            'p-value': '',
            'sampleNum': ''
        })

result_df = pd.DataFrame(results)
print(result_df)
