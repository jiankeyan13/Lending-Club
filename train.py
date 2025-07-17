import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
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

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=3407)

xgb_aucs, mlp_aucs, avg_aucs = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 识别数值型特征（不包括独热编码）
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    num_cols = [col for col in num_cols if not (set(X_train[col].unique()) <= {0, 1})]

    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight,
                        random_state=42, n_jobs=-1, max_depth=5, learning_rate=0.1, n_estimators=300)
    xgb.fit(X_train, y_train)
    xgb_val_pred = xgb.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_val_pred)
    xgb_aucs.append(xgb_auc)
    print(f"  XGBoost AUC: {xgb_auc:.4f}")

    # MLP
    scaler = MinMaxScaler()
    X_train_mlp = X_train.copy()
    X_val_mlp = X_val.copy()
    X_train_mlp[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val_mlp[num_cols] = scaler.transform(X_val[num_cols])
    X_train_mlp = X_train_mlp.apply(pd.to_numeric, errors='coerce').astype(np.float32)
    X_val_mlp = X_val_mlp.apply(pd.to_numeric, errors='coerce').astype(np.float32)
    X_train_mlp = X_train_mlp.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val_mlp = X_val_mlp.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train_tensor = torch.tensor(X_train_mlp.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_mlp.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

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
        mlp_val_pred = torch.sigmoid(mlp(X_val_tensor)).squeeze().numpy()
    mlp_auc = roc_auc_score(y_val, mlp_val_pred)
    mlp_aucs.append(mlp_auc)
    print(f"  MLP (PyTorch) AUC: {mlp_auc:.4f}")

    # 概率平均
    avg_pred = (xgb_val_pred + mlp_val_pred) / 2
    avg_auc = roc_auc_score(y_val, avg_pred)
    avg_aucs.append(avg_auc)
    print(f"  XGB+MLP 平均概率 AUC: {avg_auc:.4f}")

print("\n5折交叉验证平均AUC：")
print(f"  XGBoost: {np.mean(xgb_aucs):.4f}")
print(f"  MLP (PyTorch): {np.mean(mlp_aucs):.4f}")
print(f"  XGB+MLP 平均概率: {np.mean(avg_aucs):.4f}")
