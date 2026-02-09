import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ===============================
# 1. 原始线性规划参数
# ===============================
c = -np.array([0.473, 0.526])   # 最大化 → 取负

A_base = np.array([
    [166.7, 1800],   # e
    [4.92, 5.745],   # G
    [559, 90]        # t
])

b = np.array([1103.35, 5.57, 350])

bounds = [(0, None), (0, None)]

A_eq = np.array([[1, 1]])
b_eq = np.array([1])

# ===============================
# 2. 扰动倍率
# ===============================
ratios = np.linspace(0.9, 1.1, 21)

x1_e, x1_G, x1_t = [], [], []

def solve_lp(A):
    res = linprog(
        c,
        A_ub=A,
        b_ub=b,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )
    if res.success:
        return res.x[0]
    else:
        return np.nan

# ===============================
# 3. e 扰动
# ===============================
for r in ratios:
    A = A_base.copy()
    A[0, :] *= r
    x1_e.append(solve_lp(A))

# ===============================
# 4. G 扰动
# ===============================
for r in ratios:
    A = A_base.copy()
    A[1, :] *= r
    x1_G.append(solve_lp(A))

# ===============================
# 5. t 扰动
# ===============================
for r in ratios:
    A = A_base.copy()
    A[2, :] *= r
    x1_t.append(solve_lp(A))

# ===============================
# 6. 绘图（一张图三条线）
# ===============================
print("e :", x1_e)
print("G :", x1_G)
print("t :", x1_t)

plt.figure()
plt.plot(ratios, x1_e, label="e")
plt.plot(ratios, x1_G, label="G")
plt.plot(ratios, x1_t, label="t")
plt.xlabel("倍率")
plt.ylabel("x1 最优解")
plt.legend()
plt.grid(True)
plt.savefig("figures/LP_sensitivity_analysis.png")