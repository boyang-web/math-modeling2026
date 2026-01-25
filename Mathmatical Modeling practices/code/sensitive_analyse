import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=np.linspace(0,10,100)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

a0, b0, c0 = 1, 1, 1
deltas = [-0.2, -0.1, 0, 0.1, 0.2]

ax = axs[0, 0]

for da in deltas:
    y_var = (a0 + da) * x**2 + b0*x + c0
    ax.plot(x, y_var, label=f"a={a0+da:.1f}")

ax.set_title("Sensitivity to a")
ax.set_ylabel("y")
ax.grid(True)
ax.legend(fontsize=7)

ax = axs[0, 1]

for db in deltas:
    y_var = a0*x**2 + (b0 + db)*x + c0
    ax.plot(x, y_var)

ax.set_title("Sensitivity to b")
ax.grid(True)

ax = axs[1, 0]

for dc in deltas:
    y_var = a0*x**2 + b0*x + (c0 + dc)
    ax.plot(x, y_var)

ax.set_title("Sensitivity to c")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

axs[1, 1].axis("off")
axs[1, 1].text(
    0.1, 0.5,
    "Sensitivity Analysis\nBaseline vs Variations",
    fontsize=12
)

plt.show()