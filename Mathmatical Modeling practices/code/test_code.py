import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ourmodel(x):
    return x**2+x+1

data=pd.read_csv("data/test_data.csv")
t=data["t"].values
real=data["real"].values
given_model=data["given_model"].values

our_model=ourmodel(t)

#数据对比图
plt.figure(figsize=(8,5))

plt.scatter(t,real,label="real data",color="black",s=20)
plt.plot(t,given_model,label="given_model",color="black",linestyle="--")
plt.plot(t,our_model,label="our_model",color="red",linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("test")

plt.legend()
plt.grid(True)
plt.savefig("figures/model_compare.png", dpi=300, bbox_inches="tight")

#残差分析图
plt.figure(figsize=(8, 5))
res_given = real - given_model
res_our = real - our_model

plt.scatter(t, res_given, label="given model residual", color="black")
plt.scatter(t, res_our, label="our model residual", color="red")

plt.axhline(0, linestyle="--")  # y=0 的参考线

plt.xlabel("t")
plt.ylabel("residual")
plt.title("Residual comparison")

plt.legend()
plt.grid(True)

plt.savefig("figures/residual_compare.png", dpi=300,bbox_inches="tight")

