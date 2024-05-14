import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost

#--------------------------导入文件-------------------

# 假设从'updated_features.csv'读取数据，这里用示例数据代替实际操作
df = pd.read_csv('updated_features.csv')

#------------------------势头差转折点识别----------------------

# # 初始化diff_sig列为0
# df['diff_sig'] = 0

# # 获取数据的符号
# signs = np.sign(df['momentum_diff'])

# # 检查相邻行数据的符号变化，并根据逻辑更新diff_sig
# for i in range(len(df) - 1):
#     current_sign = signs.iloc[i]
#     next_sign = signs.iloc[i + 1]
    
#     # 如果相邻两行的数据符号相反
#     if current_sign != next_sign:
#         # 如果行数较大的数据为负，则行数较小的diff_sig为-1
#         # 如果行数较大的数据为正，则行数较小的diff_sig为1
#         df.at[i, 'diff_sig'] = -1 if next_sign < 0 else 1
#     # 如果符号相同，则diff_sig保持为0（这一步实际上可以省略，因为diff_sig已经初始化为0）

# # 显示修改后的DataFrame以供检查
# print(df)

# # 保存到CSV文件中，替换原有文件或创建新文件
# df.to_csv('updated_features.csv', index=False)

#---------------------XGBoost预测特征排序-------------------------


# # 将目标变量映射到0, 1, 2
# target_mapping = {-1: 0, 0: 1, 1: 2}
# df['mapped_diff_sig'] = df['diff_sig'].map(target_mapping)
# y = df['mapped_diff_sig']

# # 特征选择
# features = ['p1_skill_score', 'p2_skill_score', 'p1_error_score', 'p2_error_score', 'win_streak']
# X = df[features]

# # 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# # 初始化XGBoost模型并训练
# model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8)
# model.fit(X_train, y_train)

# # 特征重要性
# feature_importances = model.feature_importances_

# # 可视化特征重要性
# plt.figure(figsize=(10, 6))
# plt.barh(features, feature_importances)
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importance Ranking')
# plt.show()

#----------------------------多因素与胜率因素排序-----------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from xgboost import plot_importance

# 读取数据
data = pd.read_csv('updated_features.csv')  # 替换为你的数据文件路径

# 特征和标签选择
X = data[['p1_skill_score', 'p2_skill_score', 'p1_error_score', 'p2_error_score', 'p1_win_streak', 'p2_win_streak']]
y = data['point_victor'] = data['point_victor'].apply(lambda x: 1 if x == 1 else 0)


# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 类别不平衡处理
weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

# XGBoost参数
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# 训练模型
dtrain = xgboost.DMatrix(X_train, label=y_train, weight=weights)
dtest = xgboost.DMatrix(X_test)
bst = xgboost.train(params, dtrain, num_boost_round=100)

# 输出特征的重要性
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(bst, ax=ax, grid=False)
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()

# 预测概率
ypred = bst.predict(dtest)

# 绘制并保存每个match_id的赢得比赛的概率走势图
for match_id in data['match_id'].unique():
    match_data = data[data['match_id'] == match_id]
    match_X = match_data[X.columns]
    match_dtest = xgboost.DMatrix(match_X)
    match_ypred = bst.predict(match_dtest)
    
    plt.figure(figsize=(10, 5))
    plt.plot(match_data['elapsed_time'], match_ypred, label='Win Probability')
    plt.xlabel('Elapsed Time')
    plt.ylabel('Probability')
    plt.title(f'Match ID {match_id} Win Probability Trend')
    plt.legend()
    plt.savefig(f'match_{match_id}_win_probability_trend.png')
    plt.close()

# 保存带有赢得比赛概率的数据
data['win_probability'] = np.nan
data.loc[X_test.index, 'win_probability'] = ypred
data.to_csv('updated_with_win_probabilities.csv', index=False)
