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

#创建子图
fig,(ax1,ax2)=plt.subplots(
    2,1,
    figsize=(8,6),
    sharex=True,
    gridspec_kw={"height_ratios":[3,1]}
)

ax1.plot(x,y,color="grey",linestyle=":",label="ideal_data")
ax1.plot(x,data["y1"],color="red",label="noise_data",linewidth=1)

ax1.set_title("generated_data")
ax1.set_ylabel("generated_data")

ax1.minorticks_on()#更细的网格
ax1.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
ax1.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

ax1.legend(loc="center left",frameon=False,bbox_to_anchor=(1.02,0.8))

ax2.axhline(0,color="grey",linewidth=1)
sigma = np.std(residual)
# ±1σ
ax2.axhline(+sigma, color="blue", linestyle="--", linewidth=1, label="σ")
ax2.axhline(-sigma, color="blue", linestyle="--", linewidth=1)
# ±2σ
ax2.axhline(+2*sigma, color="purple", linestyle=":", linewidth=1, label="2σ")
ax2.axhline(-2*sigma, color="purple", linestyle=":", linewidth=1)

ax2.scatter(x,residual,color="red",s=25)
ax2.set_xlabel("x")
ax2.set_ylabel("Residual")

ax2.legend(loc="center left",frameon=False,bbox_to_anchor=(1.02,0.5))

ax2.minorticks_on()
ax2.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
ax2.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

fig.savefig("figures/generated_data.png",dpi=600,bbox_inches="tight")#防止legend放在图外被裁掉

#直方图
plt.figure(figsize=(6,4))

plt.hist(
    residual,
    bins=20,
    density=True,    
    color="grey",
    edgecolor="black",
    alpha=0.7
)

plt.axvline(0, color="black", linewidth=1)
plt.axvline(+sigma, color="blue", linestyle="--", linewidth=1)
plt.axvline(-sigma, color="blue", linestyle="--", linewidth=1)

plt.xlabel("Residual")
plt.ylabel("Density")
plt.title("Residual Histogram")

plt.grid(True, linestyle="--", alpha=0.5, label="σ")

plt.legend(loc="center left",frameon=False,bbox_to_anchor=(1.02,0.5))

plt.savefig("figures/residual_hist.png", dpi=600, bbox_inches="tight")
