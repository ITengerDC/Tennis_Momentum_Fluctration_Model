import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_C_t(win_streak, prev_win_streak):
    a = 0.5  # Larger coefficient when a winning or losing streak is broken
    b = 0.1  # Smaller coefficient when a winning or losing streak continues

    # Check if the winning or losing streak is broken
    if (win_streak < 0 and prev_win_streak > 0) or (win_streak > 0 and prev_win_streak < 0):
        coefficient = a
    else:
        coefficient = b

    # Calculate C_t with a smoother growth after 6 or 7 consecutive wins/losses
    if abs(win_streak) <= 4:
        C_t = coefficient * win_streak
    else:
        C_t = coefficient * (4 + np.sqrt(abs(win_streak) - 4))
    
    return C_t if win_streak >= 0 else -C_t

# Load the data
df = pd.read_csv('../data/updated_features.csv')

# 初始化新列以存储动量分数和动量差
df['p1_momentum_score'] = np.nan
df['p2_momentum_score'] = np.nan
df['momentum_diff'] = np.nan

# Ensure the directory for saving plots exists
plots_dir = 'momentum_difference_plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Process each match
for match_id in df['match_id'].unique():
    match_df = df[df['match_id'] == match_id].copy()
    match_df.sort_values(by='point_no', inplace=True)

    # Shift the columns to get previous point's data
    for player in ['p1', 'p2']:
        match_df[f'prev_{player}_skill_score'] = match_df[f'{player}_skill_score'].shift(1).fillna(0)
        match_df[f'prev_{player}_error_score'] = match_df[f'{player}_error_score'].shift(1).fillna(0)
        match_df[f'prev_{player}_win_streak'] = match_df[f'{player}_win_streak'].shift(1).fillna(0)

    # # Initialize momentum scores for both players
    # match_df['p1_momentum_score'] = np.nan
    # match_df['p2_momentum_score'] = np.nan

    # Calculate momentum scores
    for index, row in match_df.iterrows():
        P_t_p1 = 2 if row['point_victor'] == 1 else 1
        P_t_p2 = 2 if row['point_victor'] == 2 else 1

        S_t_p1 = 1.2 if row['server'] == 1 else 1.0
        S_t_p2 = 1.2 if row['server'] == 2 else 1.0

        # When index is 0, we only consider S_t because there is no previous point data
        if index == 0:
            match_df.at[index, 'p1_momentum_score'] = S_t_p1
            match_df.at[index, 'p2_momentum_score'] = S_t_p2
            continue

        C_t_p1 = calculate_C_t(row['p1_win_streak'], row['prev_p1_win_streak'])
        C_t_p2 = calculate_C_t(row['p2_win_streak'], row['prev_p2_win_streak'])

        # Calculate momentum scores and store them
        match_df.at[index, 'p1_momentum_score'] = P_t_p1 * S_t_p1 + C_t_p1 + row['prev_p1_skill_score'] - row['prev_p1_error_score']
        match_df.at[index, 'p2_momentum_score'] = P_t_p2 * S_t_p2 + C_t_p2 + row['prev_p2_skill_score'] - row['prev_p2_error_score']
        # Calculate the momentum difference between players
        match_df['momentum_diff'] = match_df['p1_momentum_score'] - match_df['p2_momentum_score']

    # 更新原始DataFrame
    df.update(match_df[['p1_momentum_score', 'p2_momentum_score', 'momentum_diff']])

    # Plot and save the momentum difference graph
    plt.figure()
    plt.plot(match_df['point_no'], match_df['momentum_diff'], label=f'Momentum Difference for Match {match_id}')
    plt.axhline(y=0, color='gray', linestyle='--')  # Add a horizontal line at y=0 for reference
    plt.xlabel('Point Number')
    plt.ylabel('Momentum Difference')
    plt.title(f'Match {match_id} Momentum Difference Flow')
    plt.legend()
    plot_file_path = os.path.join(plots_dir, f'match_{match_id}_momentum_difference_plot.png')
    plt.savefig(plot_file_path)
    plt.close()
# 保存更新后的数据到CSV
df.to_csv('updated_features.csv', index=False)
