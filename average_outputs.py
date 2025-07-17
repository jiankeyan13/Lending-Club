import pandas as pd

# 读取两个模型的结果
mlp_df = pd.read_csv('output_task1_mlp.csv')
xgb_df = pd.read_csv('output_task1_xgboost.csv')

# 按id合并
merged = pd.merge(mlp_df, xgb_df, on='id', suffixes=('_mlp', '_xgb'))

# 取outcome均值，保留8位小数
merged['outcome'] = ((merged['outcome_mlp'] + merged['outcome_xgb']) / 2).round(8)

# 只保留id和新的outcome
output = merged[['id', 'outcome']]
output.to_csv('output_task1.csv', index=False)
print('output_task1.csv 已生成！（MLP与XGBoost结果平均，保留8位小数）') 