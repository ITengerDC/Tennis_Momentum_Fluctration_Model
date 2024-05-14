import pandas as pd
import numpy as np
from ARIMA import ARIMA_model
from Bradley_Terry import BradleyTerryModel
from SARIMAX import SARIMAX_model
from XGBoost import XGBoost_model
from 特征组合及性能检验 import train_logistic_regression

def train_models():
    # 加载数据
    data = pd.read_csv('data/updated_features.csv')

    # 使用ARIMA模型
    arima_results = ARIMA_model(data['some_time_series_column'])

    # 使用Bradley-Terry模型
    bt_model = BradleyTerryModel(data)
    bt_results = bt_model.fit()

    # 使用SARIMAX模型
    sarimax_results = SARIMAX_model(data['some_other_time_series_column'])

    # 使用XGBoost模型
    xgb_model = XGBoost_model(data.drop(columns=['target']), data['target'])
    xgb_results = xgb_model.train()

    # 使用Logistic回归模型
    logistic_results = train_logistic_regression(data)

    # 保存模型和结果
    np.save('models/arima_results.npy', arima_results)
    np.save('models/bt_results.npy', bt_results)
    np.save('models/sarimax_results.npy', sarimax_results)
    xgb_model.save_model('models/xgb_model.json')
    np.save('models/logistic_results.npy', logistic_results)

    print("Training completed and models saved.")

if __name__ == "__main__":
    train_models()
