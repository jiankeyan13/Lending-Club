import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# 1. 读取数据
train_df = pd.read_csv('data/new_train.csv')
test_df = pd.read_csv('data/new_test.csv')

# 2. 特征和标签
X_train = train_df.drop(columns=['isDefault'])
y_train = train_df['isDefault'].astype(int)
X_test = test_df.drop(columns=['id']) if 'id' in test_df.columns else test_df.copy()
test_id = test_df['id'] if 'id' in test_df.columns else None

# 3. 识别数值型特征（不包括独热编码）
num_cols = X_train.select_dtypes(include=[np.number]).columns
num_cols = [col for col in num_cols if not (set(X_train[col].unique()) <= {0, 1})]

# 4. MLP预测
scaler = MinMaxScaler()
X_train_mlp = X_train.copy()
X_test_mlp = X_test.copy()
X_train_mlp[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_mlp[num_cols] = scaler.transform(X_test[num_cols])
X_train_mlp = X_train_mlp.apply(pd.to_numeric, errors='coerce').astype(np.float32)
X_test_mlp = X_test_mlp.apply(pd.to_numeric, errors='coerce').astype(np.float32)
X_train_mlp = X_train_mlp.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_mlp = X_test_mlp.replace([np.inf, -np.inf], np.nan).fillna(0)

def to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

X_train_tensor = to_tensor(X_train_mlp)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = to_tensor(X_test_mlp)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

mlp = MLP(X_train_tensor.shape[1])
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(mlp.parameters(), lr=0.002)

EPOCHS = 25
mlp.train()
for epoch in range(EPOCHS):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = mlp(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

mlp.eval()
with torch.no_grad():
    mlp_pred = torch.sigmoid(mlp(X_test_tensor)).squeeze().cpu().numpy()
output_mlp = pd.DataFrame({'id': test_id, 'outcome': mlp_pred})
output_mlp.to_csv('output_task1_mlp.csv', index=False)
print('output_task1_mlp.csv 已生成!（MLP概率分数）')

# 5. XGBoost预测
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight,
                   random_state=42, n_jobs=-1, max_depth=5, learning_rate=0.1, n_estimators=300)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict_proba(X_test)[:, 1]
output_xgb = pd.DataFrame({'id': test_id, 'outcome': xgb_pred})
output_xgb.to_csv('output_task1_xgb.csv', index=False)
print('output_task1_xgb.csv 已生成！（XGBoost概率分数）')

# 6. 融合（平均）
avg_pred = ((mlp_pred + xgb_pred) / 2).round(8)
output_avg = pd.DataFrame({'id': test_id, 'outcome': avg_pred})
output_avg.to_csv('output_task1.csv', index=False)
print('output_task1.csv 已生成！（MLP与XGBoost概率平均，保留8位小数）') 