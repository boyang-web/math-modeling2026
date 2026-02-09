import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("data/launchestime.csv")
x = data["year"].values
y = data["launchesnumber"].values

# 指数模型
def exp_func(x, a, b):
    return a * np.exp(b * x)

from scipy.optimize import curve_fit

# 使用对数线性回归计算稳健的初值（仅在 y>0 时）
mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
if mask.sum() >= 2:
    try:
        coef = np.polyfit(x[mask], np.log(y[mask]), 1)
        b0 = coef[0]
        a0 = np.exp(coef[1])
        p0 = (a0, b0)
    except Exception:
        p0 = (1.0, 0.01)
else:
    p0 = (1.0, 0.01)

# 尝试用 curve_fit，增加 maxfev 并设置合理边界；失败时退化为对数线性拟合
try:
    popt, pcov = curve_fit(
        exp_func,
        x,
        y,
        p0=p0,
        maxfev=10000,
        bounds=([0, -np.inf], [np.inf, np.inf])
    )
    a, b = popt
except Exception as e:
    if mask.sum() >= 2:
        coef = np.polyfit(x[mask], np.log(y[mask]), 1)
        b = coef[0]
        a = np.exp(coef[1])
    else:
        raise RuntimeError("无法拟合指数模型：数据不足或包含非正值") from e

# 生成拟合曲线并绘图
x_fit = np.linspace(2011, 2050, 40)
y_fit = exp_func(x_fit, a, b)
arr = data["launchesnumber"].tolist()
arr.extend([np.nan] * (40 - len(arr)))
data1 = pd.DataFrame({
    "x": x_fit,
    "y": y_fit,
    "z": arr
})

plt.plot(data1["x"], data1["y"], label="Fitted Exponential Model")
plt.plot(
    data1["x"], data1["z"], label="Actual Launches Number",
     linestyle='--', marker='o'
)
plt.xlabel("Year")
plt.ylabel("Number of Launches")
plt.title("Launches Over Time with Exponential Fit")
plt.legend(
    loc="center left",
    frameon=False,
    fontsize=8,
    bbox_to_anchor=(1.02, 0.85)
)
plt.minorticks_on()
plt.grid(which="major", alpha=0.7, linestyle="-", linewidth=0.5)
plt.grid(which="minor", alpha=0.4, linestyle=":", linewidth=0.3)


plt.scatter(2050, exp_func(2050, a, b), color="red", zorder=5)
#圆点上写出2050年的数值，保留到整数

plt.text(2050, exp_func(2050, a, b), f"{exp_func(2050, a, b):.0f}", verticalalignment='bottom', color='red')
#打印出拟合参数
print(f"Fitted Exponential Model Parameters: a = {a}, b = {b}")
plt.savefig("figures/launches_over_time_exponential.png", dpi=600, bbox_inches='tight')
# 再画一张只显示到2030年的图，其余与上述相同，并保存
#plt.xlim(2011, 2030)
#纵坐标显示范围改到0到1500
#plt.ylim(0, 600)
#横坐标固定为整数
#plt.xticks(np.arange(2011, 2031, 2))    
#plt.savefig("figures/launches_over_time_exponential_2030.png", dpi=600, bbox_inches='tight')
