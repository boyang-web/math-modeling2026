import numpy as np
import matplotlib.pyplot as plt

# m 的范围（由 delta ∈ [0.21, 0.55] 推出）
m = np.linspace(901.7, 1457.0, 300)

# delta 表达式
delta = (1800 - m) / 1633.3

# f(m)
f_m = delta * 550 + (1 - delta) * 90

# 画图
plt.figure(figsize=(7, 5))
plt.plot(m, f_m)
#在图上打印出解析式
plt.text(1000, 100, r"$f(m) = \frac{1800 - m}{1633.3} \cdot 550 + (1 - \frac{1800 - m}{1633.3}) \cdot 90$", fontsize=12)

plt.xlabel(r"$e_{limit}$")
plt.ylabel("t")

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
plt.savefig("figures/f_m_vs_m.png", dpi=600, bbox_inches="tight")