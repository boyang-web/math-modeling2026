import numpy as np
from scipy.optimize import linprog

c = -np.array([0.473, 0.526])   # 最大化 → 取负

A = np.array([
     [166.7, 1800],   # e
    [4.92, 5.745],   # G
    [559, 90]        # t
])
b = np.array([  1103.35,5.57, 350])
bounds = [(0, None), (0, None)]
# 再加一个限制，x1+x2=1
A_eq = np.array([[1, 1]])
b_eq = np.array([1])

    
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method="highs")

print("最优解:", res.x)
print("最大值:", -res.fun)
