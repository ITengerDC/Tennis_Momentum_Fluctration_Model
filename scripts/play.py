import sys
import os

# 将src目录添加到系统路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ARIMA import ARIMA_model
from Bradley_Terry import BradleyTerryModel
from SARIMAX import SARIMAX_model
from XGBoost import XGBoost_model
from 特征组合及性能检验 import train_logistic_regression

def play_results():
    # 加载数据
    data = pd.read_csv('data/updated_features.csv')

    # 加载模型和结果
    arima_results = np.load('models/arima_results.npy')
    bt_results = np.load('models/bt_results.npy')
    sarimax_results = np.load('models/sarimax_results.npy')
    logistic_results = np.load('models/logistic_results.npy')

    xgb_model = XGBoost_model(data.drop(columns=['target']), data['target'])
    xgb_model.load_model('models/xgb_model.json')

    # 展示ARIMA结果
    plt.figure()
    plt.plot(arima_results)
    plt.title('ARIMA Model Results')
    plt.show()

    # 展示Bradley-Terry模型结果
    print("Bradley-Terry Model Results:", bt_results)

    # 展示SARIMAX结果
    plt.figure()
    plt.plot(sarimax_results)
    plt.title('SARIMAX Model Results')
    plt.show()

    # 展示XGBoost结果
    xgb_predictions = xgb_model.predict(data.drop(columns=['target']))
    plt.figure()
    plt.plot(xgb_predictions)
    plt.title('XGBoost Model Predictions')
    plt.show()

    # 展示Logistic回归结果
    print("Logistic Regression Results:", logistic_results)

    print("Analysis completed and results displayed.")

if __name__ == "__main__":
    play_results()
