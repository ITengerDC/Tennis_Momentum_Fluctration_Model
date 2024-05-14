import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import os

# 加载数据
df = pd.read_csv('updated_features.csv')

# 势头差数据预处理
# 确保数据是按时间顺序排列的，这里假设df已经是时间序列数据

# 定义一个函数来检查时间序列的平稳性
def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    if result[1] <= 0.05:
        print("Series is stationary")
        return True
    else:
        print("Series is not stationary")
        return False

# 检查势头差的平稳性
if not check_stationarity(df['momentum_diff']):
    print("Consider differencing or transformation to make the series stationary before using SARIMAX.")
    # 在这里可以进行差分或转换
    # df['momentum_diff'] = df['momentum_diff'].diff().dropna()

# 假设经过上述步骤后，我们确认了使用SARIMAX是合适的

# 定义模型参数
order = (1, 0, 1)  # 根据ACF和PACF图调整
seasonal_order = (0, 0, 0, 0)  # 如果检测到季节性，调整这里

# 准备外生变量
exog = df[['p1_skill_score', 'p2_skill_score', 'p1_error_score', 'p2_error_score', 'win_streak']]

# 定义SARIMAX模型并拟合
model = SARIMAX(df['momentum_diff'], exog=exog, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

# 进行预测
df['predicted_momentum_diff'] = results.predict(start=0, end=len(df)-1, exog=exog)
# 计算预测误差（实际值与预测值之间的差）
df['pred_error'] = df['predicted_momentum_diff'] - df['momentum_diff']

# 计算预测误差的绝对值
df['abs_error'] = np.abs(df['predicted_momentum_diff'] - df['momentum_diff'])

# 确定覆盖90%和95%散点的误差阈值
error_threshold_90 = df['abs_error'].quantile(0.90)
error_threshold_95 = df['abs_error'].quantile(0.95)

# 输出两条带的宽度
print(f"Band width for 90% of points: {2 * error_threshold_90}")
print(f"Band width for 95% of points: {2 * error_threshold_95}")

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['momentum_diff'], df['predicted_momentum_diff'], alpha=0.5, label='Data Points')

# 绘制y=x线
x_vals = np.linspace(df['momentum_diff'].min(), df['momentum_diff'].max(), 100)
plt.plot(x_vals, x_vals, '-r', label='Ideal Prediction (y=x)')

# 绘制90%和95%误差带
plt.fill_between(x_vals, x_vals - error_threshold_90, x_vals + error_threshold_90, color='yellow', alpha=0.3, label='90% Error Band')
plt.fill_between(x_vals, x_vals - error_threshold_95, x_vals + error_threshold_95, color='green', alpha=0.3, label='95% Error Band')

plt.title('Momentum Gap Prediction with Error Bands')
plt.xlabel('Actual Momentum Gap')
plt.ylabel('Predicted Momentum Gap')
plt.legend()
plt.show()
