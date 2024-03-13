import pandas as pd
import numpy as np
import os
import torch.optim as optim
import torch.nn.functional as F
from visualization import plotly_graph
from utils import add_features, create_time_dataset
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    train_df = pd.read_csv(
        "/home/jinjiarui/run/Time-Series-Learning/LSTM-Temperature-Forcast/data/DailyDelhiClimateTrain.csv"
    )
    test_df = pd.read_csv(
        "/home/jinjiarui/run/Time-Series-Learning/LSTM-Temperature-Forcast/data/DailyDelhiClimateTest.csv"
    )
    # 特征工程
    train_df = add_features(train_df)
    test_df = add_features(test_df)
    print(train_df.head())

    # 提取数据
    train_variables = train_df[
        [
            "month",
            "day",
            "humidity",
            "wind_speed",
            "meanpressure",
            "humidity_pressure_ratio",
            "meantemp",
        ]
    ].values.astype("float32")
    test_variables = test_df[
        [
            "month",
            "day",
            "humidity",
            "wind_speed",
            "meanpressure",
            "humidity_pressure_ratio",
            "meantemp",
        ]
    ].values.astype("float32")
    concat_all_data = np.concatenate((train_variables, test_variables), axis=0)
    print(train_variables.shape, test_variables.shape, concat_all_data.shape)

    # 绘制图像
    graph_data = train_df[["meantemp", "humidity", "wind_speed", "meanpressure"]].values
    graph = plotly_graph(graph_data, train_df["date"])
    graph.box_plot()
    graph.line_plot()

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_variables = scaler.fit_transform(train_variables)
    test_variables = scaler.fit_transform(test_variables)
    concat_all_data = scaler.fit_transform(concat_all_data)

    # 构建数据集
    feature_index = 6  # 选取meantemp作为单变量时间序列
    lookback = 7  # 回溯时间步
    forcast_lengths = 7  # 预测长度
    train_data, train_target = create_time_dataset(train_variables, feature_index, lookback, forcast_lengths)
    test_data, test_target = create_time_dataset(test_variables, feature_index, lookback, forcast_lengths)
