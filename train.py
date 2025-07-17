import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 读取数据
DATA_PATH = 'data/new_train.csv'
df = pd.read_csv(DATA_PATH)

# 2. 划分特征和标签
X = df.drop(columns=['isDefault'])
y = df['isDefault'].astype(int)

# 3. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 识别数值型特征（不包括独热编码）
num_cols = X_train.select_dtypes(include=[np.number]).columns
# 只保留0/1列以外的数值型特征
num_cols = [col for col in num_cols if not (set(X_train[col].unique()) <= {0, 1})]

# 5. 随机森林
rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_val_pred = rf.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_val_pred)

# 6. XGBoost
# 计算 scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
xgb_val_pred = xgb.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_val_pred)

# 7. MLP (PyTorch)
# 只对数值型特征做MinMaxScaler
scaler = MinMaxScaler()
X_train_mlp = X_train.copy()
X_val_mlp = X_val.copy()
X_train_mlp[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val_mlp[num_cols] = scaler.transform(X_val[num_cols])

# 转为Tensor
def to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

X_train_tensor = to_tensor(X_train_mlp)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = to_tensor(X_val_mlp)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

mlp = MLP(X_train_tensor.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

# 训练MLP
EPOCHS = 20
mlp.train()
for epoch in range(EPOCHS):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = mlp(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

# 验证MLP
mlp.eval()
with torch.no_grad():
    mlp_val_pred = mlp(X_val_tensor).squeeze().numpy()
mlp_auc = roc_auc_score(y_val, mlp_val_pred)

# 8. 绘制ROC曲线
plt.figure(figsize=(8,6))
for name, y_score in zip([
    'Random Forest', 'XGBoost', 'MLP (PyTorch)'],
    [rf_val_pred, xgb_val_pred, mlp_val_pred]):
    fpr, tpr, _ = roc_curve(y_val, y_score)
    plt.plot(fpr, tpr, label=f'{name}')
plt.plot([0,1],[0,1],'k--',label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation Set)')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 输出AUC分数
print(f'Random Forest AUC: {rf_auc:.4f}')
print(f'XGBoost AUC: {xgb_auc:.4f}')
print(f'MLP (PyTorch) AUC: {mlp_auc:.4f}')
