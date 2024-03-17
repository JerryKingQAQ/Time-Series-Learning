import pandas as pd
import numpy as np
import os
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visualization import plotly_graph
from utils import add_features, average_predictions, create_time_dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from model import BiLSTM, LinearModel
from train import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_df = pd.read_csv(
        "/home/jinjiarui/run/Time-Series-Learning/LSTM-Temperature-Forecast/data/DailyDelhiClimateTrain.csv"
    )
    test_df = pd.read_csv(
        "/home/jinjiarui/run/Time-Series-Learning/LSTM-Temperature-Forecast/data/DailyDelhiClimateTest.csv"
    )
    # 特征工程
    train_df = add_features(train_df)
    test_df = add_features(test_df)
    print(train_df.head())

    # 提取数据
    train_variables = train_df[
        [
            "meantemp",
            "humidity",
            "wind_speed",
            "meanpressure",
            "humidity_pressure_ratio",
            "month",
            "day",
        ]
    ].values.astype("float32")
    test_variables = test_df[
        [
            "meantemp",
            "humidity",
            "wind_speed",
            "meanpressure",
            "humidity_pressure_ratio",
            "month",
            "day",
        ]
    ].values.astype("float32")
    concat_variables = np.concatenate((train_variables, test_variables), axis=0)
    print(
        "Data Shape: ",
        train_variables.shape,
        test_variables.shape,
        concat_variables.shape,
    )

    concat_time = np.array(range(train_variables.shape[0] + test_variables.shape[0]))

    # 绘制图像
    graph = plotly_graph(concat_variables[:, 2:6], concat_time)
    graph.box_plot()
    graph.line_plot()

    # 选择特征列
    feature_index = 0  # 选取meantemp作为单变量时间序列
    train_feature = train_variables[:, feature_index].reshape(-1, 1)
    test_feature = test_variables[:, feature_index].reshape(-1, 1)
    concat_feature = np.concatenate((train_feature, test_feature), axis=0)
    print(
        "Feature Shape: ", train_feature.shape, test_feature.shape, concat_feature.shape
    )

    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = RobustScaler()
    # scaler = StandardScaler()
    train_feature = scaler.fit_transform(train_feature)
    test_feature = scaler.fit_transform(test_feature)
    concat_feature = scaler.fit_transform(concat_feature)

    # 构建数据集
    lookback = 7  # 回溯时间步
    forecast_lengths = 2  # 预测长度
    step = 1 # 步长
    train_data, train_target = create_time_dataset(
        train_feature, lookback, forecast_lengths, device, step
    )
    test_data, test_target = create_time_dataset(
        test_feature, lookback, forecast_lengths, device, step
    )

    print("Train Dataset shape: ", train_data.shape, train_target.shape)
    print("Test Dataset shape: ", test_data.shape, test_target.shape)

    train_loader = DataLoader(
        TensorDataset(train_data, train_target), batch_size=512, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_data, test_target), batch_size=512, shuffle=False
    )

    # 构建模型 优化器 损失函数
    model = BiLSTM(
        input_size=lookback, hidden_size=256, num_layers=2, output_size=forecast_lengths
    ).to(device)
    # model = LSTM(
    #     input_size=lookback, hidden_size=256, num_layers=2, output_size=forecast_lengths
    # ).to(device)
    # model = LinearModel(
    #     input_size=lookback, hidden_size=256, output_size=forecast_lengths
    # ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=10)
    loss = nn.MSELoss()

    # 训练模型
    best_train_preds, best_test_preds = train(
        model, train_loader, test_loader, loss, optimizer, scheduler, n_epochs=100
    )

    # print("Best Test Preds: ", best_test_preds)

    # 对best_train_preds和best_test_preds进行平均化处理
    avg_train_preds = average_predictions(
        best_train_preds, train_variables.shape[0], forecast_lengths, lookback
    )
    avg_test_preds = average_predictions(
        best_test_preds, test_variables.shape[0], forecast_lengths, lookback
    )

    # print("Best Avg Test Preds: ", avg_test_preds)

    # 重塑avg_train_preds和avg_test_preds为二维数组
    avg_train_preds_reshaped = avg_train_preds.reshape(-1, 1)
    avg_test_preds_reshaped = avg_test_preds.reshape(-1, 1)

    # 然后对这些二维数组进行反归一化处理
    avg_train_preds_inverse = scaler.inverse_transform(avg_train_preds_reshaped)
    avg_test_preds_inverse = scaler.inverse_transform(avg_test_preds_reshaped)

    # 如果需要，可以再次将这些二维数组转换回一维数组
    avg_train_preds_final = avg_train_preds_inverse.flatten()
    avg_test_preds_final = avg_test_preds_inverse.flatten()

    # print("Final Avg Test Preds: ", avg_test_preds_final)

    # 绘制预测图像
    graph.draw_pred_pic(avg_train_preds_final, avg_test_preds_final, feature_index=0)
