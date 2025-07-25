# 任务描述 - 2025 北邮2组 考核1

信贷业务是现代金融体系的重要组成部分，银行和金融机构通过向个人和企业提供贷款来获取收益。然而，贷款违约（即借款人未能按时偿还贷款本息）是信贷业务中常见的风险之一。违约不仅会导致金融机构的资金损失，还可能影响其声誉和市场信心。因此，准确预测贷款人是否会违约对于金融机构的风险管理和决策至关重要。现在某金融机构提供过往贷款人的行为和违约情况，希望寻求有效的方法以更加准确地预测出贷款人是否会出现违约。

**结果提交：**
- 提交一个output_task1.csv文件，要求包含两列，每一列内容分别为贷款人信息id以及违约预测结果，注意id要和测试集test.csv的顺序保持一致，且要有表头id, outcome。

# 项目结构

```
Lending-Club/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── new_train.csv
│   └── new_test.csv
├── EDA/
│   └── test.ipynb
├── fig_res/
│   ├── Figure_1.png
│   └── Figure_2.png
├── data_pre.py
├── train.py
├── predict.py
└── README.md
```

# EDA（探索性数据分析）

1. **目标变量分析**：查看违约与未违约样本的比例，判断数据集是否平衡。
2. **特征相关性分析**：绘制数值特征相关性热力图，发现与违约强相关的特征。
3. **分组特征分析**：
   - 贷款属性（如金额、期限、利率）与违约关系
   - 信用评级与违约关系
   - 财务状况（年收入、债务收入比等）与违约关系
   - 就业年限、负面信用记录、信贷使用情况等与违约关系
   - 贷款用途和申请类型与违约关系

# 数据预处理（data_pre.py 实现内容）

- **特征统一**：合并训练集和测试集，保证特征列完全一致。
- **异常值处理**：数值型特征用IQR法检测并截断异常值（排除伪装成数值的分类特征）。
- **缺失值处理**：数值型用中位数填充，分类型用众数填充。
- **特征工程**：
  - term 二值化（3→0, 5→1）
  - purpose 独热编码
  - grade 有序编码（A-G 映射为 0-6）
  - employmentLength 文本转数字
  - 其他伪数值型分类特征不做异常值处理
- **特征创造**：
  - 财务健康比率（如 annualIncome/loanAmnt、installment/(annualIncome/12) 等）
  - 信用与行为整合（如 FICO 均值、活跃账户占比）
  - 风险因子交互项（如 dti*revolUtil、interestRate/信用分均值）
- **特征筛选**：删除无用或冗余特征：id、policyCode、installment、ficoRangeLow、ficoRangeHigh、pubRecBankruptcies、employmentTitle、interestRate、dti、income_div_loanAmnt 等
- **输出**：处理后的 new_train.csv 和 new_test.csv，特征顺序和数量完全一致，便于后续建模。

# 特征构建

![特征相关性](fig_res/Figure_1.png)

![特征重要性](fig_res/Figure_2.png)

# 训练与评估（train.py）

- **交叉验证**：采用5折StratifiedKFold交叉验证，保证评估结果稳定可靠。
- **模型**：
  - XGBoost（max_depth=5, learning_rate=0.1, n_estimators=300, scale_pos_weight自动处理类别不平衡）
  - MLP（PyTorch实现，含Dropout，BCEWithLogitsLoss+pos_weight处理不平衡，数值特征归一化）
  - 融合：XGBoost和MLP概率分数平均
- **评估指标**：AUC-ROC（软标签概率分数，官方评测标准）
- **输出**：每折AUC和平均AUC，便于模型对比和调优

# 预测与提交（predict.py）

- **predict.py**：集成MLP、XGBoost、概率平均三种预测方法。
  - 生成 output_task1_mlp.csv（MLP概率分数）
  - 生成 output_task1_xgb.csv（XGBoost概率分数）
  - 生成 output_task1.csv（MLP与XGBoost概率平均，推荐提交）
- **提交格式**：output_task1.csv，包含id和outcome两列，outcome为概率分数，顺序与new_test.csv一致

# 代码文件说明

- data_pre.py：数据清洗、特征工程、特征创造、特征筛选
- train.py：交叉验证训练与AUC评估，支持多模型和融合
- predict.py：MLP、XGBoost、概率平均三种预测与提交
- EDA/test.ipynb：探索性数据分析与可视化



