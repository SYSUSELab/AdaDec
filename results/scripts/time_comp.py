import pandas as pd

# 创建MBPP+原始数据
mbpp_data = {
    'tradition': [202.35, 386.81, 223.59, 3786.92, 904.30, 1544.67, 1075.00],
    'adapt': [2292.40, 2886.17, 1395.54, 3152.65, 1938.31, 2342.49, 2205.36],
    'fixl': [365.19, 492.04, 287.72, 1093.44, 860.88, 1274.97, 1514.69]
}
mbpp_index = ['ds1.3b', 'ds6.7b', 'stb3b', 'qw3-0.6b', 'qw3-1.7b', 'qw3-4b', 'qw3-8b']
mbpp_df = pd.DataFrame(mbpp_data, index=mbpp_index)

# 创建HumanEval+原始数据
humaneval_data = {
    'tradition': [188.26, 388.41, 277.98, 1020.84, 1413.88, 887.30, 883.21],
    'adapt': [396.16, 603.40, 494.65, 667.46, 693.17, 1241.66, 775.35],
    'fixl': [238.72, 564.00, 337.05, 861.25, 863.70, 1322.22, 1138.00]
}
humaneval_df = pd.DataFrame(humaneval_data, index=mbpp_index)

# 进行除法运算
mbpp_processed = mbpp_df / 378
humaneval_processed = humaneval_df / 164

# 设置显示格式，保留两位小数
pd.set_option('display.float_format', '{:.2f}'.format)

# 输出结果
print("处理后的MBPP+表格（除以378）：")
print(mbpp_processed)
print("\n处理后的HumanEval+表格（除以164）：")
print(humaneval_processed)
