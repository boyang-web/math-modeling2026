import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("data/launchestime.csv")
coef_quad=[1.92202650e+00 ,-7.74327397e+03 , 7.79891373e+06]
print("二次多项式系数：",coef_quad)
x=np.linspace(2011,2050,40)
y_quad = np.polyval(coef_quad, x)
arr =data["launchesnumber"].tolist()
arr.extend([np.nan] * (40 - len(arr)))
data1=pd.DataFrame({
    "x":x,
    "y":y_quad,
    "z":arr
})
plt.plot(data1["x"], data1["y"], label="Fitted Quadratic Model")
plt.plot(data1["x"], data1["z"], label="Actual Launches Number", linestyle='--', marker='o')
plt.xlabel("Year")
plt.ylabel("Number of Launches")
plt.title("Launches Over Time with Quadratic Fit")
plt.legend(
        loc="center left",
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(1.02,0.85)
        )
    
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
#在2050年的数据打一个点，并标上2518,写在点的左边
#plt.scatter(2050, np.polyval(coef_quad, 2050), color='red', zorder=5)
#plt.text(2050, np.polyval(coef_quad, 2050), ' 2518', verticalalignment='bottom', color='red')
#plt.savefig("figures/launches_over_time.png",dpi=600,bbox_inches='tight')

plt.xlim(2011, 2030)
#纵坐标显示范围改到0到1500
plt.ylim(0, 600)
#横坐标固定为整数
plt.xticks(np.arange(2011, 2031, 2))    
plt.savefig("figures/launches_over_time_exponential_2030.png", dpi=600, bbox_inches='tight')
