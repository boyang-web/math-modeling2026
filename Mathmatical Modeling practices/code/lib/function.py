import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def r2(y, y_fit):#相关系数计算
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot

def plot_multiline_from_csv(data,x_label,y_label,title,savename):#首列为自变量，其他列画折线图

    x = data.iloc[:, 0]
    y_cols = data.columns[1:]
    fig, ax = plt.subplots()        
    for col in y_cols:
        idx = np.argsort(x)#排序模块
        x_sorted = x[idx]
        y_sorted = data[col][idx]
        ax.plot(x_sorted, y_sorted, label=col)

    ax.set_xlabel(x_label)#设置横轴
    ax.set_ylabel(y_label)#设置纵轴
    ax.set_title(title)#设置标题

    ax.legend(
        loc="center left",
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(1.02,0.85)
        )
    
    ax.minorticks_on()
    ax.grid(
	    True,   
	    which="major",  
	    linestyle="--", 
	    linewidth=0.5,  
	    alpha=0.6
    )

    ax.grid(
        True, 
	    which="minor",
	    linestyle=":",
	    linewidth=0.3, 
	    alpha=0.4
    )

    ax.ticklabel_format(style='plain', axis='y')#此句关闭科学计数法
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))#此句保证横轴为整数

    ax.text(
        0.05, 0.95,
        f"words",#调整文本
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top'
    )   

    plt.tight_layout()
    plt.savefig(savename,dpi=600, bbox_inches="tight")

def two_model_fitting_graph(data,x_label,y1_label,y2_label,savename):#双模型拟合图
    
    x=data[data.columns[0]]
    y=data[data.columns[1]]

    fig,axs=plt.subplots(
        1,2,    
        figsize=(12,5), 
        sharex=False,  
	    gridspec_kw={ 
	    "width_ratios":[1,1],
	    }
    )

    axs[0].scatter(x,y,label="real data",color="grey",s=10)
    axs[1].scatter(x,y,label="real data",color="grey",s=10)

    #线性拟合
    coef_linear = np.polyfit(x, y, 1)
    print(coef_linear)
    y_linear = np.polyval(coef_linear, x)
    axs[0].plot(x,y_linear,label="linear matching",color="red")
    axs[0].legend(frameon=False,fontsize=8)
    axs[0].set_xlabel(x_label)#改参数显示
    axs[0].set_ylabel(y1_label)
    axs[0].minorticks_on()
    axs[0].grid(
	    True, 
	    which="major", 
	    linestyle="--", 
	    linewidth=0.5, 
	    alpha=0.6
    )

    axs[0].grid(
    True, 
	    which="minor",
	    linestyle=":",
	    linewidth=0.3, 
	    alpha=0.4
    )

    r2_value=r2(y,y_linear)

    axs[0].text(
     0.05, 0.95,
     f"$R^2 = {r2_value:.3f}$",
    transform=axs[0].transAxes,
    fontsize=12,
    verticalalignment='top'
    )
    #二次拟合
    coef_quad = np.polyfit(x, y, 2)
    print(coef_quad)
    y_quad = np.polyval(coef_quad, x)
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y_quad[idx]
    axs[1].plot(x_sorted, y_sorted,label="quad matching",color="red")
    axs[1].legend(frameon=False,fontsize=8)
    axs[1].set_xlabel(x_label)#改参数显示
    axs[1].set_ylabel(y2_label)
    axs[1].minorticks_on()
    axs[1].grid(
	    True, 
	    which="major", 
	    linestyle="--", 
	    linewidth=0.5, 
	    alpha=0.6
    )

    axs[1].grid(
     True, 
	    which="minor",
	    linestyle=":",
	    linewidth=0.3, 
	    alpha=0.4
    )

    r2_value=r2(y,y_quad)

    axs[1].text(
        0.05, 0.95,
     f"$R^2 = {r2_value:.3f}$",
     transform=axs[1].transAxes,
        fontsize=12,
        verticalalignment='top'
    )


    plt.tight_layout()
    plt.savefig(savename,dpi=600, bbox_inches="tight")

def residual_analysis(data,savename):#输入三列数据，x,y,y1分析残差
   
    #filename需要满足要求
    x=data[data.columns[0]]
    y=data[data.columns[1]]
    y1=data[data.columns[2]]
    residual=y1-y

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(8, 8),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]}
        )

    ax1.plot(x, y, color="grey", linestyle=":", label=data.columns[1])
    ax1.plot(x, y1, color="red", label=data.columns[2])
    ax1.set_ylabel("y")#改参数显示
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
    fig.savefig(savename, dpi=600, bbox_inches="tight")

def plot_heatmap(data, title,savename):#输入带有行列名的csv，产出热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True, # 数据写在格子里
        cmap="YlOrRd", #配色方案,viridis,YlOrRd
        square=True  #正方形格
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savename, dpi=600, bbox_inches="tight")
