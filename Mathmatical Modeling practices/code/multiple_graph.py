import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=np.linspace(0,10,100)
noise=np.random.normal(0,2,size=len(x))
y=x**2+x+1
data=pd.DataFrame({
    "x":x,
    "y":y,
    "y1":y+noise
})
data.to_csv("data/generated_data.csv",index=False)
residual=noise#残差

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1,
    figsize=(8, 8),
    sharex=False,
    gridspec_kw={"height_ratios": [3, 1.5, 1.5]}
)

ax1.plot(x, y, color="grey", linestyle=":", label="ideal")
ax1.plot(x, data["y1"], color="red", label="noisy data")
ax1.set_ylabel("y")
ax1.set_title("Model & Residual Analysis")

ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(frameon=False)

sigma = np.std(residual)

ax2.axhline(0, color="black")
ax2.axhline(+sigma, color="blue", linestyle="--", label="±1σ")
ax2.axhline(-sigma, color="blue", linestyle="--")
ax2.axhline(+2*sigma, color="purple", linestyle=":", label="±2σ")
ax2.axhline(-2*sigma, color="purple", linestyle=":")

ax2.scatter(x, residual, color="red", s=20)
ax2.set_ylabel("Residual")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(frameon=False, fontsize=8)

ax3.hist(
    residual,
    bins=20,
    density=True,
    orientation="horizontal",
    color="grey",
    edgecolor="black",
    alpha=0.7
)

ax3.axhline(0, color="black")
ax3.axhline(+sigma, color="blue", linestyle="--")
ax3.axhline(-sigma, color="blue", linestyle="--")

ax3.set_xlabel("Density")
ax3.set_ylabel("Residual")
ax3.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig("figures/full_residual_analysis.png", dpi=600)

