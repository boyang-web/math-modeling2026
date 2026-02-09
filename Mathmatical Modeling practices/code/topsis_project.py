import pandas as pd
import numpy as np

def topsis(df, weights=None, impacts=None):
    """
    df: DataFrame, 行为方案，列为指标
    weights: list or np.array, 每个指标权重，默认等权
    impacts: list, 每个指标影响方向，'+'为越大越好，'-'为越小越好，默认全部正向
    """
    data = df.values.astype(float)
    n_samples, n_features = data.shape

    # 默认权重
    if weights is None:
        weights = np.ones(n_features) / n_features
    else:
        weights = np.array(weights)
    
    # 默认影响方向
    if impacts is None:
        impacts = ['+'] * n_features

    # 1. 数据标准化（向量归一化）
    norm_data = data / np.sqrt((data**2).sum(axis=0))

    # 2. 加权
    weighted_data = norm_data * weights

    # 3. 找理想解和负理想解
    ideal = np.zeros(n_features)
    anti_ideal = np.zeros(n_features)
    for j in range(n_features):
        if impacts[j] == '+':
            ideal[j] = weighted_data[:, j].max()
            anti_ideal[j] = weighted_data[:, j].min()
        else:
            ideal[j] = weighted_data[:, j].min()
            anti_ideal[j] = weighted_data[:, j].max()

    # 4. 计算距离
    dist_ideal = np.sqrt(((weighted_data - ideal)**2).sum(axis=1))
    dist_anti = np.sqrt(((weighted_data - anti_ideal)**2).sum(axis=1))

    # 5. 计算TOPSIS得分
    score = dist_anti / (dist_ideal + dist_anti)

    return score

# ===============================
# 使用示例
# ===============================
if __name__ == "__main__":
    # 读取你的数据（假设已保存为CSV，也可以直接创建DataFrame）
    data = pd.DataFrame({
        "e": [166.7, 1800, 983.35],
        "G": [4.92E12, 5.745E12, 5.3325E12],
        "t": [559, 90, 324.5]
    }, index=["Elevator", "Rocket", "optimize"])

    # 指定权重和影响方向（这里示例全等权，全部正向）
    weights = [1, 1, 1]  # 可以替换为熵权法计算的权重
    impacts = ['+', '+', '+']  # '+' 越大越好，'-' 越小越好

    # 计算TOPSIS得分
    scores = topsis(data, weights, impacts)

    # 保存到DataFrame
    data['score'] = scores

    print(data)

    # 保存到CSV
    data.to_csv("topsis_scores.csv", encoding="utf-8-sig")
