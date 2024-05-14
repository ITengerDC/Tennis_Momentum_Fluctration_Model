import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 确保在您的系统上安装了Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 23  # 可以根据需要调整字体大小
plt.rcParams['font.weight'] = 'bold'

#--------------------------特征与结果的相关性分析---------------------------------
# 加载数据
df = pd.read_csv('updated_features.csv')

# 特征和目标变量
X = df[['p1_skill_score', 'p2_skill_score', 'p1_error_score', 'p2_error_score', 'win_streak']]
y = df['point_victor']

# # 1. 逻辑回归分析
# log_reg = LogisticRegression()
# log_reg.fit(X, y)

# # 获取特征的概率估计
# probability_estimates = pd.DataFrame(log_reg.predict_proba(X), columns=log_reg.classes_)
# print("概率估计：\n", probability_estimates)

# 2. 相关系数
correlation_matrix = X.corr()
print("相关系数矩阵：\n", correlation_matrix)
# 对列名进行处理，用换行符替换下划线
columns = [col.replace('_', '\n') for col in correlation_matrix.columns]

plt.figure(figsize=(12, 10))  # 可以根据实际需求调整画布大小
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',  fmt='0.2f',
            annot_kws={'weight': 'bold'},xticklabels=columns, yticklabels=columns)
plt.title('Feature Correlation Matrix', weight='bold')
plt.xticks(rotation=0, weight='bold')  # 可以根据需要调整角度
plt.yticks(rotation=0, weight='bold')   # 保持y轴标签的默认角度
plt.savefig('feature_correlation_matrix.png')
plt.close()  # 关闭当前图形，以免影响后续图形的绘制

# # 3. GBDT 分析
# gbdt = GradientBoostingClassifier()
# gbdt.fit(X, y)

# # 绘制第一棵决策树并保存
# plt.figure(figsize=(20, 10))
# plot_tree(gbdt.estimators_[0, 0], filled=True, feature_names=X.columns, proportion=True)
# plt.title('Decision Tree from GBDT')
# plt.savefig('decision_tree_from_gbdt.png')
# plt.close()  # 关闭当前图形

#------------------------------势头差与胜率的相关性分析------------------------------
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
# from scipy.stats import pointbiserialr
# # 假设df是你的DataFrame，其中包含势头差（'momentum_diff'）和胜率（'win_rate'，值为1或2）
# df = pd.read_csv('updated_features.csv')

# #-------逻辑回归--------

# # 将胜率的值转换为0和1，这是逻辑回归分析所需的（1表示类别1，0表示类别2）
# df['win_binary'] = np.where(df['point_victor'] == 1, 1, 0)
# # 定义特征和标签
# X = df[['momentum_diff']]  # 特征
# y = df['win_binary']       # 标签
# # 拆分数据为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# # 创建逻辑回归模型
# log_reg = LogisticRegression()
# # 训练模型
# log_reg.fit(X_train, y_train)
# # 预测测试集
# y_pred = log_reg.predict(X_test)
# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
# # 查看分类报告
# print(classification_report(y_test, y_pred))

# #-------点双列相关系数--------

# # 计算点双列相关系数
# corr, p_value = pointbiserialr(df['win_binary'], df['momentum_diff'])
# print(f"Point-Biserial Correlation coefficient: {corr}")
# print(f"P-value: {p_value}")
