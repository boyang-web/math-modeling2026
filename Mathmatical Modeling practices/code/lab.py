import pandas as pd
import numpy as np
from lib.function import two_model_fitting_graph
from lib.function import residual_analysis
from lib.function import plot_heatmap
from lib.function import plot_multiline_from_csv
#residual_analysis("data/test_data.csv","figures/save1.png")

#data=pd.read_csv("data/testheap.csv",index_col=0)
#plot_heatmap(data,"testheat","figures/save2.png")

#data=pd.read_csv("data/launchestime.csv")
#two_model_fitting_graph(data,"x","y","y","figures/save3.png")

#coef_quad=[ 1.92202650e+00 ,-7.74327397e+03 , 7.79891373e+06]
#x=np.linspace(2011,2050,40)
#y_quad = np.polyval(coef_quad, x)



#arr =data["launchesnumber"]
#arr.extend([np.nan] * (40 - len(arr)))

#data1=pd.DataFrame({
 #   "x":x,
  #  "y":y_quad,
  #  "z":arr
#})
#plot_multiline_from_csv(data1,"x","y","title","figures/save4.png")

#import numpy as np
#import pandas as pd

# 原始数据
#data = pd.DataFrame([
   # [166.7, 4.92e12, 559],
  #  [1800,  5.745e12, 90],
  #  [983.35, 5.3325e12, 324.5]
#], columns=["e", "G", "t"])

# 向量标准化（按列）
#data_norm = data / np.sqrt((data ** 2).sum(axis=0))

#print(1-data_norm)

#data=pd.read_csv("data/failrate.csv")
#two_model_fitting_graph(data,"x","y","y","figures/failrate_fit.png")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ===============================
# 1. 基础参数
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

ratios = [0.95, 1.0, 1.05]

# ===============================
# 2. 求解函数
# ===============================
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
# 3. 敏感性计算
# ===============================
results = {
    "e": [],
    "G": [],
    "t": []
}

for r in ratios:
    # e 扰动
    A = A_base.copy()
    A[0, :] *= r
    results["e"].append(solve_lp(A))

    # G 扰动
    A = A_base.copy()
    A[1, :] *= r
    results["G"].append(solve_lp(A))

    # t 扰动
    A = A_base.copy()
    A[2, :] *= r
    results["t"].append(solve_lp(A))

# ===============================
# 4. 点状图
# ===============================
plt.figure()

plt.scatter(ratios, results["e"], label="e", s=60)
plt.scatter(ratios, results["G"], label="G", s=60)
plt.scatter(ratios, results["t"], label="t", s=60)

plt.xlabel("倍率")
plt.ylabel("x1 最优解")
plt.legend()
plt.grid(True)
plt.savefig("figures/LP_sensitivity_analysis.png")
