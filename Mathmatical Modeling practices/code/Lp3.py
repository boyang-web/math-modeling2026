import pandas as pd
import numpy as np

def topsis_score(df, weights, impacts):
    data = df.values.astype(float)

    # 向量归一化
    norm = data / np.sqrt((data**2).sum(axis=0))

    # 加权
    weighted = norm * weights

    # 理想解 / 负理想解
    ideal = np.zeros(weighted.shape[1])
    anti = np.zeros(weighted.shape[1])

    for j in range(weighted.shape[1]):
        if impacts[j] == '+':
            ideal[j] = weighted[:, j].max()
            anti[j] = weighted[:, j].min()
        else:
            ideal[j] = weighted[:, j].min()
            anti[j] = weighted[:, j].max()

    d_pos = np.sqrt(((weighted - ideal)**2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti)**2).sum(axis=1))

    return d_neg / (d_pos + d_neg)

import matplotlib.pyplot as plt

ratios = np.linspace(0.9, 1.1, 21)

# LP 给出的分配系数（示例）
x1, x2 = 0.3, 0.7

weights = np.array([1, 1, 1])
impacts = ['+', '+', '+']

results = {"e": [], "G": [], "t": []}

for r in ratios:
    # ========= e 扰动 =========
    data_e = pd.DataFrame({
        "e": [166.7*r, 1800*r],
        "G": [4.92, 5.745],
        "t": [559, 90]
    }, index=["Elevator", "Rocket"])

    score_e = topsis_score(data_e, weights, impacts)
    total_e = x1*score_e[0] + x2*score_e[1]
    results["e"].append(total_e)

    # ========= G 扰动 =========
    data_G = pd.DataFrame({
        "e": [166.7, 1800],
        "G": [4.92*r, 5.745*r],
        "t": [559, 90]
    }, index=["Elevator", "Rocket"])

    score_G = topsis_score(data_G, weights, impacts)
    total_G = x1*score_G[0] + x2*score_G[1]
    results["G"].append(total_G)

    # ========= t 扰动 =========
    data_t = pd.DataFrame({
        "e": [166.7, 1800],
        "G": [4.92, 5.745],
        "t": [559*r, 90*r]
    }, index=["Elevator", "Rocket"])

    score_t = topsis_score(data_t, weights, impacts)
    total_t = x1*score_t[0] + x2*score_t[1]
    results["t"].append(total_t)

plt.figure()

plt.plot(ratios, results["e"], label="e disturbance")
plt.plot(ratios, results["G"], label="G disturbance")
plt.plot(ratios, results["t"], label="t disturbance")

plt.xlabel("Disturbance ratio")
plt.ylabel("Overall evaluation factor")
plt.legend()
plt.grid(True)

plt.savefig("figures/LP_TOPSIS_sensitivity_analysis.png")