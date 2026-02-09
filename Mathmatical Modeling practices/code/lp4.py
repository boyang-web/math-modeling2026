import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ===============================
# 1. TOPSIS 函数（支持极小指标）
# ===============================
def topsis(df, weights, impacts):
    data = df.values.astype(float)
    # 向量归一化
    norm = data / np.sqrt((data**2).sum(axis=0))
    # 加权
    weighted = norm * weights
    n, m = weighted.shape

    ideal = np.zeros(m)
    anti = np.zeros(m)
    for j in range(m):
        if impacts[j] == '+':  # 越大越好
            ideal[j] = weighted[:, j].max()
            anti[j] = weighted[:, j].min()
        else:  # '-' 越小越好
            ideal[j] = weighted[:, j].min()
            anti[j] = weighted[:, j].max()

    d_pos = np.sqrt(((weighted - ideal)**2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti)**2).sum(axis=1))
    score = d_neg / (d_pos + d_neg)
    return score

# ===============================
# 2. 初始数据
# ===============================
data_base = pd.DataFrame({
    "e": [166.7, 1800, 1103.35],
    "G": [4.92, 5.745, 5.5735],
    "t": [559, 90, 350]
}, index=["Elevator", "Rocket", "Reference"])

weights = np.array([1, 1, 1])
# 极小指标：'-' 表示越小越好
impacts = ['-', '-', '-']  

# LP 约束
A = np.array([
    [166.7, 1800],
    [4.92, 5.745],
    [559, 90]
])
b = np.array([1103.35, 5.57, 350])
bounds = [(0, None), (0, None)]
A_eq = np.array([[1, 1]])
b_eq = np.array([1])

# 扰动倍率
ratios = np.linspace(0.9, 1.1, 21)

# ===============================
# 3. 总评价指标计算函数
# ===============================
def compute_total_evaluation(target_scheme):
    results = {"e": [], "G": [], "t": []}
    for r in ratios:
        for indicator in ["e", "G", "t"]:
            # 复制基础数据
            df_perturb = data_base.copy()
            # 只扰动目标方案的一个指标
            df_perturb.loc[target_scheme, indicator] *= r
            # TOPSIS 得分
            scores = topsis(df_perturb, weights, impacts)
            # LP 目标函数系数
            c_lp = -np.array([scores[0], scores[1]])
            # 线性规划求分配系数
            res = linprog(c_lp, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.success:
                x1, x2 = res.x
            else:
                x1, x2 = np.nan, np.nan
            # 总评价指标
            total_eval = x1*scores[0] + x2*scores[1]
            results[indicator].append(total_eval)
    return results

# ===============================
# 4. 计算火箭扰动和电梯扰动
# ===============================
rocket_results = compute_total_evaluation("Rocket")
elevator_results = compute_total_evaluation("Elevator")

# ===============================
# 5. 绘图
# ===============================
# 火箭扰动图
plt.figure()
plt.plot(ratios, rocket_results["e"], 'o-', label="e disturbance")
plt.plot(ratios, rocket_results["G"], 's-', label="G disturbance")
plt.plot(ratios, rocket_results["t"], '^-', label="t disturbance")
plt.xlabel("Rocket disturbance ratio")
plt.ylabel("Total evaluation factor")
plt.title("Total evaluation vs Rocket disturbance (min indicators)")
plt.legend(
        loc="center left",
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(1.02,0.85)
        )
    
plt.minorticks_on()
plt.grid(
	    True,   
	    which="major",  
	    linestyle="--", 
	    linewidth=0.5,  
	    alpha=0.6
    )

plt.grid(
        True, 
	    which="minor",
	    linestyle=":",
	    linewidth=0.3, 
	    alpha=0.4
    )
plt.tight_layout()
plt.savefig("figures/LP3_sensitivity_analysis_rocket_min_indicators.png", dpi=600, bbox_inches='tight')

# 电梯扰动图
plt.figure()
plt.plot(ratios, elevator_results["e"], 'o-', label="e disturbance")
plt.plot(ratios, elevator_results["G"], 's-', label="G disturbance")
plt.plot(ratios, elevator_results["t"], '^-', label="t disturbance")
plt.xlabel("Elevator disturbance ratio")
plt.ylabel("Total evaluation factor")
plt.title("Total evaluation vs Elevator disturbance (min indicators)")
plt.legend(
        loc="center left",
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(1.02,0.85)
        )
    
plt.minorticks_on()
plt.grid(
	    True,   
	    which="major",  
	    linestyle="--", 
	    linewidth=0.5,  
	    alpha=0.6
    )

plt.grid(
        True, 
	    which="minor",
	    linestyle=":",
	    linewidth=0.3, 
	    alpha=0.4
    )
plt.tight_layout()
plt.savefig("figures/LP3_sensitivity_analysis_min_indicators.png", dpi=600, bbox_inches='tight')
