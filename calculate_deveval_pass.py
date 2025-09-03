import pandas as pd
import numpy as np

# 创建三个表格的数据
data_0_100 = {
    'model': ['ds1.3b', 'ds6.7b', 'stb3b', 'clm7b', 'qw3-0.6b', 'qw3-1.7b', 'qw3-4b', 'qw3-8b'],
    'tradition': [15, 23, 11, 15, 8, 11, 13, 7],
    'fixl': [17, 22, 18, 19, 11, 15, 17, 15],
    'dynl': [18, 22, 19, 17, 9, 13, 15, 15]
}

data_100_500 = {
    'model': ['ds1.3b', 'ds6.7b', 'stb3b', 'clm7b', 'qw3-0.6b', 'qw3-1.7b', 'qw3-4b', 'qw3-8b'],
    'tradition': [9, 15.75, 8.75, 11.25, 4, 6, 6.75, 6],
    'fixl': [13, 16.75, 13.75, 16.75, 5.5, 9, 10.75, 9.5],
    'dynl': [11.75, 16.25, 13, 17, 6.25, 8.5, 9.25, 9.75]
}

data_501_1825 = {
    'model':        ['ds1.3b',  'ds6.7b',   'stb3b',    'clm7b',    'qw3-0.6b', 'qw3-1.7b', 'qw3-4b',   'qw3-8b'],
    'tradition':    [1.66,      0,          1.51,       0,          1.21,          0,          0,          0],
    'fixl':         [2.04,         0,          1.81,          0,          1.36,          0,          0,          0],
    'dynl':         [0,         0,          0,          0,          0,          0,          0,          0]
}

# 转换为DataFrame
df_0_100 = pd.DataFrame(data_0_100)
df_100_500 = pd.DataFrame(data_100_500)
df_501_1825 = pd.DataFrame(data_501_1825)

# 题目数量
n_0_100 = 100
n_100_500 = 400  # 500 - 100
n_501_1825 = 1325  # 1825 - 500
total_questions = 1825

# 计算加权平均正确率
df_total = df_0_100.copy()
for col in ['tradition', 'fixl', 'dynl']:
    df_total[col] = (
        df_0_100[col] * n_0_100 +
        df_100_500[col] * n_100_500 +
        df_501_1825[col] * n_501_1825
    ) / total_questions

# 设置显示格式，保留2位小数
df_total = df_total.round(2)

# 输出结果
print("1825道题目的总正确率表格：")
print(df_total)