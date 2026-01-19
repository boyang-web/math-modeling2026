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

plt.figure(figsize=(8,5))

plt.scatter(t,real,label="real data",color="black")
plt.plot(t,given_model,label="given_model",color="black",linestyle="--")
plt.plot(t,our_model,label="our_model",color="red")

plt.xlabel("x")
plt.ylabel("y")
plt.title("test")

plt.legend()
plt.grid(True)
plt.savefig("figures/model_compare.png", dpi=300, bbox_inches="tight")
plt.show()
