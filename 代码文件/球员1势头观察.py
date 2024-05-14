import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_C_t(win_streak, prev_win_streak):
    a = 0.5  # 当连胜或连败被打破时的较大系数
    b = 0.1  # 当连胜或连败未被打破时的较小系数

    # 检查连胜或连败是否被打破
    if (win_streak < 0 and prev_win_streak > 0) or (win_streak > 0 and prev_win_streak < 0):
        coefficient = a
    else:
        coefficient = b

    # 计算C_t
    if abs(win_streak) <= 4:
        C_t = coefficient * win_streak
    else:
        C_t = coefficient * (4 + np.sqrt(abs(win_streak) -4))
    
    return C_t if win_streak >= 0 else -C_t

# 加载数据
df = pd.read_csv('updated_features.csv')

# 确保用于保存图表的文件夹存在
plots_dir = 'momentum_plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

for match_id in df['match_id'].unique():
    match_df = df[df['match_id'] == match_id].copy()
    match_df.sort_values(by='point_no', inplace=True)  # 确保按比赛进行的时间排序

    # 使用shift()方法创建上一个得分点的技术得分和失误分列
    match_df['prev_p1_skill_score'] = match_df['p1_skill_score'].shift(1).fillna(0)  # 第一行没有前一行，用0填充
    match_df['prev_p1_error_score'] = match_df['p1_error_score'].shift(1).fillna(0)  # 同上
    # 使用shift()方法创建上一个得分点的技术得分和失误分列
    match_df['prev_win_streak'] = match_df['p1_win_streak'].shift(1).fillna(0)
    # 初始化势头分数列
    match_df['momentum_score'] = np.nan

    momentum_score = 0

    for index, row in match_df.iterrows():
        P_t = 2 if row['point_victor'] == row['server'] else 1
        S_t = 1.2 if row['server'] == 1 else 1.0
        # 当index为0时，将C_t赋0
        C_t = calculate_C_t(row['p1_win_streak'], row['prev_win_streak']) if index > 0 else 0
        # 计算势头分数
        momentum_score = P_t * S_t + C_t + row['prev_p1_skill_score'] + row['prev_p1_error_score']
        match_df.at[index, 'momentum_score'] = momentum_score

    # 绘制并保存势头分数变化图
    plt.figure()
    plt.plot(match_df['point_no'], match_df['momentum_score'], label=f'Momentum Score for Match {match_id}')
    plt.xlabel('Point Number')
    plt.ylabel('Momentum Score')
    plt.title(f'Match {match_id} Momentum Flow')
    plt.legend()
    plot_file_path = os.path.join(plots_dir, f'match_{match_id}_momentum_plot.png')
    plt.savefig(plot_file_path)
    plt.close()

print("Momentum plots saved to folder:", plots_dir)

