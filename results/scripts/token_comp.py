import pandas as pd

data = {
    'Model': ['DS-1.3B', 'DS-6.7B', 'ST-3B', 'CL-7B', 'QW-0.6B', 'QW-1.7B', 'QW-4B', 'QW-8B'],
    'HumanEval': {
        'Greedy':     [10869, 10549, 13116, 17207, 27653, 16493, 18915, 15620],
        'Beam':       [12267, 11099, 12009, 17162, 16188, 12066, 14696, 13424],
        'AppFiveL':   [10558, 10185, 11509, 14766, 17637, 13430, 18346, 13906],
        'AppDynL':    [10920, 9964, 11168, 14503, 16416, 13678, 14609, 14613]
    },
    'MBPP': {
        'Greedy':     [15122, 14050, 14366, 18175, 68214, 26360, 17310, 18917],
        'Beam':       [14221, 12437, 12623, 16206, 16187, 17090, 10499, 12050],
        'AppFiveL':   [10898, 11855, 10931, 13642, 34242, 17276, 16544, 15668],
        'AppDynL':    [11872, 10709, 10706, 13723, 30890, 19731, 14844, 14393]
    }
}

df_human = pd.DataFrame(data['HumanEval'], index=data['Model'])
df_mbpp = pd.DataFrame(data['MBPP'], index=data['Model'])

df_human_avg = df_human / 164
df_mbpp_avg = df_mbpp / 200

def calc_delta(df):
    base = df['Greedy']
    return df.apply(lambda col: (col - base) / base * 100)

delta_human = calc_delta(df_human_avg)
delta_mbpp = calc_delta(df_mbpp_avg)

avg_delta = pd.DataFrame({
    'HumanEval': delta_human.mean(),
    'MBPP': delta_mbpp.mean()
})

print("=== Avg Token (HumanEval) ===")
print(df_human_avg.round(2))
print("\n=== Avg Token (MBPP) ===")
print(df_mbpp_avg.round(2))
print("\n===  Delta (HumanEval) ===")
print(delta_human.round(2))
print("\n=== Delta (MBPP) ===")
print(delta_mbpp.round(2))
print("\n=== Avg Delta ===")
print(avg_delta.round(2))
