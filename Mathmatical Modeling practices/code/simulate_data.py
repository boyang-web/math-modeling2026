import numpy as np
import pandas as pd

# 固定随机种子，保证可复现（非常重要！）
np.random.seed(42)

# 原始数据
data = {
    "year": [2020, 2021, 2022, 2023, 2024],
    "sales_tax": [np.nan, np.nan, 53890397, np.nan, np.nan],
    "hotel_tax": [np.nan, np.nan, 2583590, np.nan, np.nan],
    "infrastructure": [np.nan, np.nan, 269265010, np.nan, np.nan],
    "community": [np.nan, np.nan, 3036665, np.nan, np.nan]
}

df = pd.DataFrame(data)

growth_rate = 0.10          # 年增长率 10%
fluctuation = 0.03          # 随机波动 ±3%

# 从 2022 向前、向后生成
base_year = 2022
base_index = df[df["year"] == base_year].index[0]

for col in ["sales_tax", "hotel_tax", "infrastructure", "community"]:
    
    # 向后生成（2023, 2024）
    for i in range(base_index + 1, len(df)):
        eps = np.random.uniform(-fluctuation, fluctuation)
        df.loc[i, col] = df.loc[i-1, col] * (1 + growth_rate) * (1 + eps)
    
    # 向前生成（2021, 2020）
    for i in range(base_index - 1, -1, -1):
        eps = np.random.uniform(-fluctuation, fluctuation)
        df.loc[i, col] = df.loc[i+1, col] / ((1 + growth_rate) * (1 + eps))

# 四舍五入，符合财务数据习惯
df.iloc[:, 1:] = df.iloc[:, 1:].round(0)

df.to_csv("data/simulate.csv",index=False)
