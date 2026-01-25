
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

  

data=pd.read_csv("data/test_data.csv")


t=data["t"]

real=data["real"]

model=data["given_model"]

minus=real-model

  

plt.figure(figsize=(8,5))

plt.scatter(t,real,label="real data",color="black")

plt.plot(t,model,label="model data",linestyle="--")

plt.plot(t,minus,label="difference",color="black")

plt.xlabel("x")

plt.ylabel("y")

plt.title("test 1")

plt.legend()

plt.grid(True)

plt.show()