import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# 1. 读取数据
train_df = pd.read_csv('data/new_train.csv')
test_df = pd.read_csv('data/new_test.csv')

# 2. 特征和标签
X_train = train_df.drop(columns=['isDefault'])
y_train = train_df['isDefault'].astype(int)
X_test = test_df.drop(columns=['id']) if 'id' in test_df.columns else test_df.copy()
test_id = test_df['id'] if 'id' in test_df.columns else None

# 3. XGBoost配置
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight,
                   random_state=42, n_jobs=-1, max_depth=5, learning_rate=0.1, n_estimators=300)

# 4. 训练
xgb.fit(X_train, y_train)

# 5. 预测概率分数
xgb_test_pred = xgb.predict_proba(X_test)[:, 1]

# 6. 生成提交文件（概率分数）
output = pd.DataFrame({'id': test_id, 'outcome': xgb_test_pred})
output.to_csv('output_task1.csv', index=False)
print('output_task1.csv 已生成！（XGBoost概率分数，适用于AUC评估）') 