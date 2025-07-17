import pandas as pd
import re

def preprocess_outliers_and_missing(df):
    # 处理数值型特征异常值和缺失值
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        # IQR异常值检测
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
        # 缺失值用中位数填充
        median = df[col].median()
        df[col] = df[col].fillna(median)
    # 处理分类型特征缺失值
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0])
        else:
            df[col] = df[col].fillna('missing')
    return df

def remove_useless_features(df):
    # 删除非特征或冗余无信息列
    drop_cols = []

    # policyCode 无信息
    if 'policyCode' in df.columns:
        drop_cols.append('policyCode')
    # installment 与 loanAmnt 冗余
    if 'installment' in df.columns:
        drop_cols.append('installment')
    # FICO分区间已合并为均值
    if 'ficoRangeLow' in df.columns:
        drop_cols.append('ficoRangeLow')
    if 'ficoRangeHigh' in df.columns:
        drop_cols.append('ficoRangeHigh')
    # pubRecBankruptcies 与 pubRec 冗余
    if 'pubRecBankruptcies' in df.columns:
        drop_cols.append('pubRecBankruptcies')
    if 'employmentTitle' in df.columns:
        drop_cols.append('employmentTitle')
    df = df.drop(columns=drop_cols)
    return df

def preprocess_clean(df):

    # 1. term: 3->0, 5->1
    if 'term' in df.columns:
        df['term'] = df['term'].map({3: 0, 5: 1})
    # 2. homeOwnership: 保持原样
    # 3. purpose: 独热编码
    if 'purpose' in df.columns:
        purpose_dummies = pd.get_dummies(df['purpose'], prefix='purpose')
        df = pd.concat([df.drop('purpose', axis=1), purpose_dummies], axis=1)
    # 4. applicationType: 保持原样
    # 5. policyCode: 删除（已在 remove_useless_features 处理）
    # 6. grade: 有序编码
    if 'grade' in df.columns:
        grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        df['grade'] = df['grade'].map(grade_map)
    # 7. employmentLength: 文本转数字
    if 'employmentLength' in df.columns:
        df['employmentLength'] = df['employmentLength'].replace({'10+ years': '10 years', '< 1 year': '0 years'})
        df['employmentLength'] = df['employmentLength'].str.extract(r'(\d+)').astype(float).fillna(0).astype(int)
    
    return df

def create_features(df):
    import numpy as np
    # 1. 财务健康比率
    if set(['annualIncome', 'loanAmnt']).issubset(df.columns):
        df['income_div_loanAmnt'] = np.where(df['loanAmnt'] != 0, df['annualIncome'] / df['loanAmnt'], np.nan)
        df['loanAmnt_div_income'] = np.where(df['annualIncome'] != 0, df['loanAmnt'] / df['annualIncome'], np.nan)
    if set(['installment', 'annualIncome']).issubset(df.columns):
        df['installment_div_income'] = np.where(df['annualIncome'] != 0, df['installment'] / (df['annualIncome'] / 12), np.nan)
    # 2. 信用与行为整合
    if set(['ficoRangeLow', 'ficoRangeHigh']).issubset(df.columns):
        df['fico_avg'] = (df['ficoRangeLow'] + df['ficoRangeHigh']) / 2
    if set(['openAcc', 'totalAcc']).issubset(df.columns):
        df['acc_open_ratio'] = np.where(df['totalAcc'] != 0, df['openAcc'] / df['totalAcc'], np.nan)
    # 3. 风险因子交互
    if set(['dti', 'revolUtil']).issubset(df.columns):
        df['dti_x_revolUtil'] = df['dti'] * df['revolUtil']
    if 'interestRate' in df.columns and 'fico_avg' in df.columns:
        df['interestRate_div_fico'] = np.where(df['fico_avg'] != 0, df['interestRate'] / df['fico_avg'], np.nan)
    return df

def save_data(df):
    # 保存处理后的数据
    df.to_csv('data/new_train.csv', index=False)
    print('数据预处理完成，已保存到 data/new_train.csv')

def unify_train_test(train, test):
    # 标记来源
    train['is_train'] = 1
    test['is_train'] = 0
    if 'isDefault' not in test.columns:
        test['isDefault'] = None
    all_data = pd.concat([train, test], ignore_index=True)
    # 预处理异常值和缺失值
    all_data = preprocess_outliers_and_missing(all_data)
    # 特征工程
    all_data = preprocess_clean(all_data)
    # 特征创造
    all_data = create_features(all_data)
    # 删除无用特征（但不删id）
    all_data = remove_useless_features(all_data)
    # 拆分
    new_train = all_data[all_data['is_train'] == 1].drop(['is_train'], axis=1)
    new_test = all_data[all_data['is_train'] == 0].drop(['is_train', 'isDefault'], axis=1)
    # 只对new_train删除id，new_test保留id
    if 'id' in new_train.columns:
        new_train = new_train.drop(columns=['id'])
    return new_train, new_test

if __name__ == '__main__':
    import pandas as pd
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    new_train, new_test = unify_train_test(train, test)
    new_train.to_csv('data/new_train.csv', index=False)
    new_test.to_csv('data/new_test.csv', index=False)
    print('新train和test已保存到 data/new_train.csv 和 data/new_test.csv')
