import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# 确保存储图表的文件夹存在
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# 加载数据
df = pd.read_csv('../data/updated_features.csv')

# 准备数据，计算特征差异和比赛结果编码
df['outcome'] = df['point_victor'].apply(lambda x: 1 if x == 1 else 0)

# 初始化实力参数列
df['lambda1'] = np.nan
df['lambda2'] = np.nan

# 特征列名称
feature_cols = ['p1_skill_score', 'p2_skill_score',  'p1_error_score', 'p2_error_score', 
                'p1_win_streak', 'p2_win_streak']

# 定义对数似然函数
def log_likelihood(betas, features, outcome):
    # 计算预测概率
    logits = np.dot(features, betas)
    prob = 1 / (1 + np.exp(-logits))
    # 计算对数似然
    likelihood = outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)
    return -np.sum(likelihood)

# 初始化回归系数
initial_betas = np.zeros(len(feature_cols))

# 循环每个match_id
for match_id in df['match_id'].unique():
    match_df = df[df['match_id'] == match_id].copy()
    betas_estimates = []

    # 初始化每位选手的实力参数
    lambda1 = lambda2 = 0

    # 循环每个得分点
    for index, row in match_df.iterrows():
        # 提取当前得分点的特征和结果
        features = row[feature_cols].values
        outcome = row['outcome']
        
        # 估计当前得分点的betas
        result = minimize(log_likelihood, initial_betas, args=(features, outcome), method='L-BFGS-B')
        betas = result.x
        betas_estimates.append(betas)
        
        # 更新实力参数
        lambda1 = lambda1 + betas[0] - betas[1]  # 这里的更新规则取决于模型的具体定义
        lambda2 = lambda2 + betas[1] - betas[0]
        
        # 记录实力参数估计
        df.loc[index, 'lambda1'] = lambda1
        df.loc[index, 'lambda2'] = lambda2
    
    # 绘制并保存betas走势图
    betas_df = pd.DataFrame(betas_estimates, columns=[f'beta_{i}' for i in range(len(feature_cols))])
    plt.figure()
    for beta_col in betas_df.columns:
        plt.plot(betas_df.index, betas_df[beta_col], label=beta_col)
    plt.xlabel('Point Index')
    plt.ylabel('Beta Estimate')
    plt.title(f'Match ID {match_id} Betas Over Points')
    plt.legend()
    plt.savefig(f'{plots_dir}/match_{match_id}_betas_plot.png')
    plt.close()

    # 绘制并保存实力参数走势图
    plt.figure()
    plt.plot(match_df.index, match_df['lambda1'], label='Player 1 Strength')
    plt.plot(match_df.index, match_df['lambda2'], label='Player 2 Strength')
    plt.xlabel('Point Index')
    plt.ylabel('Strength Estimate')
    plt.title(f'Match ID {match_id} Strength Over Points')
    plt.legend()
    plt.savefig(f'{plots_dir}/match_{match_id}_strength_plot.png')
    plt.close()

# 将更新后的DataFrame保存到新的CSV文件
df.to_csv('updated_with_strength_estimates.csv', index=False)
