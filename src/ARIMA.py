import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os

# 定义输入和输出文件夹路径
input_dir = r'../results/match_results'
output_dir = r'../results/arima_results'

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        
        # 加载数据
        df = pd.read_csv(file_path)
        
        # 平稳性检验
        result = adfuller(df['predicted'])
        print(f'Processing {filename}')
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        
        # ARIMA模型拟合
        model = ARIMA(df['predicted'], order=(1, 1, 1))
        model_fit = model.fit()
        
        # 绘制预测结果
        df['arima_pred'] = model_fit.predict(start=0, end=len(df) - 1, dynamic=False)
        plt.figure(figsize=(10, 6))
        plt.plot(df['predicted'], label='Original')
        plt.plot(df['arima_pred'], label='ARIMA Predicted', color='red')
        plt.title(f'ARIMA Model Predicted vs Original for {filename}')
        plt.xlabel('Index')
        plt.ylabel('Predicted Value')
        plt.legend()
        
        # 保存图表和数据
        plot_file_path = os.path.join(output_dir, f'{filename}_arima_plot.png')
        data_file_path = os.path.join(output_dir, f'{filename}_arima_data.csv')
        plt.savefig(plot_file_path)
        plt.close()
        df.to_csv(data_file_path, index=False)

print("ARIMA analysis completed and saved to", output_dir)
