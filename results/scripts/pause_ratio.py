import numpy as np
import pandas as pd

# \appFiveL
appFiveL = [
    [6.54, 5.53, 4.76, 3.95, 6.10, 11.07, 7.95, 5.98],  # HumanEval
    [8.32, 5.82, 6.35, 5.20, 6.33, 11.41, 10.64, 8.41]  # MBPP
]

# \appDynL
appDynL = [
    [6.69, 5.60, 4.72, 3.95, 5.18, 10.06, 9.37, 6.39],  # HumanEval
    [8.28, 5.89, 6.51, 5.45, 6.21, 11.88, 10.18, 7.96]  # MBPP
]

appFiveL_arr = np.array(appFiveL)
appDynL_arr = np.array(appDynL)

columns = ['DS-1.3B', 'DS-6.7B', 'ST-3B', 'CL-7B', 'QW-0.6B', 'QW-1.7B', 'QW-4B', 'QW-8B']
index = ["HumanEval", "MBPP"]

df_fiveL = pd.DataFrame(appFiveL, index=index, columns=columns)
df_dynL = pd.DataFrame(appDynL, index=index, columns=columns)

df_fiveL["Mean"] = df_fiveL.mean(axis=1)
df_dynL["Mean"] = df_dynL.mean(axis=1)

df_fiveL["Method"] = "\\appFiveL"
df_dynL["Method"] = "\\appDynL"

df_all = pd.concat([df_fiveL, df_dynL])

df_all = df_all[["Method"] + columns + ["Mean"]]

overall_mean = np.array(appFiveL + appDynL).mean()

print(df_all)
print(f"\nOverall Mean: {overall_mean:.2f}%")
