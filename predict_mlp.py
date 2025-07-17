import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

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

# 4. MinMaxScaler
scaler = MinMaxScaler()
X_train_mlp = X_train.copy()
X_test_mlp = X_test.copy()
X_train_mlp[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_mlp[num_cols] = scaler.transform(X_test[num_cols])
X_train_mlp = X_train_mlp.apply(pd.to_numeric, errors='coerce').astype(np.float32)
X_test_mlp = X_test_mlp.apply(pd.to_numeric, errors='coerce').astype(np.float32)
X_train_mlp = X_train_mlp.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_mlp = X_test_mlp.replace([np.inf, -np.inf], np.nan).fillna(0)

# 5. 转为Tensor
def to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

X_train_tensor = to_tensor(X_train_mlp)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = to_tensor(X_test_mlp)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 6. 定义MLP
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

mlp = MLP(X_train_tensor.shape[1])
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(mlp.parameters(), lr=0.002)

# 7. 训练MLP
EPOCHS = 25
mlp.train()
for epoch in range(EPOCHS):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = mlp(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

# 8. 预测并生成提交文件
mlp.eval()
with torch.no_grad():
    test_pred = torch.sigmoid(mlp(X_test_tensor)).squeeze().cpu().numpy()
outcome = (test_pred >= 0.5).astype(int)
output = pd.DataFrame({'id': test_id, 'outcome': outcome})
output.to_csv('output_task1.csv', index=False)
print('output_task1.csv 已生成！') 