import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] # 或者你系统里的任何中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 请替换成你的数据加载方式 ---
# 假设你的数据已经加载到名为 new_train 的DataFrame中
# 为了让代码可以运行，我在这里创建一个与你描述类似的虚拟数据
features = [
    'loanAmnt','term','grade','employmentLength','homeOwnership',
    'annualIncome','delinquency_2years','openAcc','pubRec','revolBal',
    'revolUtil','totalAcc','applicationType','isDefault',
    'loanAmnt_div_income','installment_div_income','fico_avg','acc_open_ratio',
    'dti_x_revolUtil','interestRate_div_fico'
]
# 创建一个随机的DataFrame作为示例
np.random.seed(42)
data = pd.read_csv('data/new_train.csv')
new_train = pd.DataFrame(data, columns=features)
# -----------------------------

# 1. 计算相关性矩阵
corr_matrix = new_train.corr()

# 2. 绘制热力图
plt.figure(figsize=(24, 20)) # 设置一个足够大的画布
sns.heatmap(
    corr_matrix, 
    annot=True,      # 在格子上显示数值
    fmt='.1f',       # 数值格式化，保留一位小数，让画面更简洁
    cmap='coolwarm', # 使用冷暖色调，红色正相关，蓝色负相关
    linewidths=.5    # 在格子之间添加细线
)
plt.title('新数据集特征相关性热力图', fontsize=20)
plt.show()

# 假设 new_train DataFrame 已经存在

# 1. 计算所有特征与 'isDefault' 的相关性
corr_with_target = new_train.corr()['isDefault'].sort_values(ascending=False)

# 2. 移除 'isDefault' 自身（它与自己的相关性总是1）
corr_with_target = corr_with_target.drop('isDefault')

# 3. 绘制条形图
plt.figure(figsize=(12, 14)) # 设置一个高一些的画布
sns.barplot(x=corr_with_target.values, y=corr_with_target.index, palette='vlag')
plt.title('各特征与目标变量(isDefault)的相关性', fontsize=16)
plt.xlabel('相关系数', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()