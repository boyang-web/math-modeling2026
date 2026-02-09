import numpy as np
import pandas as pd

def entropy_weight_method(data: pd.DataFrame):
    # 数据标准化
    #data_norm = (data - data.min()) / (data.max() - data.min())
    #data_norm = data_norm.fillna(0)

    # 比例
    p = data/ data.sum(axis=0)

    # 熵值
    epsilon = 1e-12
    k = 1.0 / np.log(len(data))
    entropy = -k * (p * np.log(p + epsilon)).sum(axis=0)

    # 熵权
    d = 1 - entropy
    weights = d / d.sum()

    return weights

# ===============================
# 使用示例
# ===============================
if __name__ == "__main__":
    data = pd.DataFrame({
        "e": [0.918993,0.125303,0.522148],
        "G": [0.468370,0.379225,0.423798],
        "t": [0.143421,0.862089,0.502755]
    })

    # 计算权重
    weights = entropy_weight_method(data)
    print("各指标权重：", weights)

    # 转成 DataFrame
    weights_df = pd.DataFrame(weights, index=data.columns, columns=["weight"])
    
    # 保存到 CSV
    weights_df.to_csv("data/entropy_weights.csv", encoding="utf-8-sig")
    print("权重已保存到 entropy_weights.csv")
