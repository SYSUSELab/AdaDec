import numpy as np

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

appFiveL_mean = appFiveL_arr.mean()
appDynL_mean = appDynL_arr.mean()

print(f"\\appFiveL mean: {appFiveL_mean:.2f}%")
print(f"\\appDynL mean: {appDynL_mean:.2f}%")

all_values = np.array(appFiveL + appDynL)

overall_mean = all_values.mean()

print(f"overall mean: {overall_mean:.2f}%")