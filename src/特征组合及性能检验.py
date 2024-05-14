import pandas as pd
import numpy as np
#-----------------------------------特征组合--------------------------------------
# # 加载数据文件
# df = pd.read_csv('new_file.csv')
# # 转换elapsed_time到秒，并计算对数
# df['elapsed_time'] = pd.to_timedelta(df['elapsed_time']).dt.total_seconds()
# df['log_elapsed_time'] = np.log1p(df['elapsed_time'])
# # 定义权重
# weights = [0.49183874, 0.3055706, 0.12479256, 0.0777981]
# # 首先计算原始的体能消耗得分
# df['p1_physical_raw'] = df['rally_count'] * df['p1_distance_run'] + df['log_elapsed_time']
# df['p2_physical_raw'] = df['rally_count'] * df['p2_distance_run'] + df['log_elapsed_time']
# # 计算特征得分
# def calculate_features(row):
#     p1_skill_score = sum([row['p1_ace'] * weights[0], row['p1_break_pt_won'] * weights[1], row['p1_winner'] * weights[2], row['p1_net_pt_won'] * weights[3]])
#     p2_skill_score = sum([row['p2_ace'] * weights[0], row['p2_break_pt_won'] * weights[1], row['p2_winner'] * weights[2], row['p2_net_pt_won'] * weights[3]])
#     p1_error_score = row['p1_unf_err']
#     p2_error_score = row['p2_unf_err']

#     return pd.Series([p1_skill_score, p2_skill_score, p1_error_score, p2_error_score])
# # 应用函数并创建新列
# df[['p1_skill_score', 'p2_skill_score', 'p1_error_score', 'p2_error_score']] = df.apply(calculate_features, axis=1)
# # 归一化体能消耗得分
# max_physical_score = max(df['p1_physical_raw'].max(), df['p2_physical_raw'].max())
# min_physical_score = min(df['p1_physical_raw'].min(), df['p2_physical_raw'].min())
# df['p1_physical_score'] = (df['p1_physical_raw'] - min_physical_score) / (max_physical_score - min_physical_score)
# df['p2_physical_score'] = (df['p2_physical_raw'] - min_physical_score) / (max_physical_score - min_physical_score)
# # 选择需要保存到新CSV的列
# columns_to_save = ['match_id', 'player1', 'player2', 'elapsed_time', 'p1_skill_score', 'p2_skill_score', 'p1_physical_score', 'p2_physical_score', 'p1_error_score', 'p2_error_score']
# new_df = df[columns_to_save]
# # 保存到新的CSV文件
# new_df.to_csv('updated_features.csv', index=False)
# print("新的特征已保存到 updated_features.csv")

#---------------------------------------特征性能验证---------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 假设df是包含上述特征和目标变量的DataFrame
df = pd.read_csv('../data/updated_features.csv')
print('point_victor' in df.columns)  # 应该输出True，如果输出False，则说明列不存在或列名有误
print(df.head())  # 查看DataFrame的前几行，确认数据是否按预期加载

#可视化验证
features = ['p1_skill_score', 'p2_skill_score', 'p1_physical_score', 'p2_physical_score', 'p1_error_score', 'p2_error_score']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='point_victor', y=feature, data=df)
    plt.title(f'Relationship between {feature} and point_victor')
    plt.savefig(f'{feature}_boxplot.png')  # 保存图片
    plt.show()

# ANOVA测试
for feature in features:
    groups = df.groupby('point_victor')[feature].apply(list)
    f_value, p_value = stats.f_oneway(*groups)
    print(f'{feature}: F值={f_value}, P值={p_value}')


