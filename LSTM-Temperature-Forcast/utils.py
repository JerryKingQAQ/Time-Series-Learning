import numpy as np
import torch


def add_features(df):
    df = add_humidity_pressure_ratio(df)
    df = add_split_time(df)
    return df


def add_humidity_pressure_ratio(df):  # 湿度压力比
    df["humidity_pressure_ratio"] = df["humidity"] / df["meanpressure"]
    return df


def add_split_time(df):  # 将'date'列切割为年、月、日，并分别添加到新的列
    split_columns = df["date"].str.split("-", expand=True)
    df["year"] = split_columns[0]
    df["month"] = split_columns[1]
    df["day"] = split_columns[2]

    return df


def create_time_dataset(data, feature_index, lookback, forcast_lengths):
    X, y = [], []
    print(np.array(data).shape)
    print(np.array(data[:, feature_index]).shape)
    for i in range(
        len(data) - lookback - forcast_lengths + 1
    ):  # feature_index = 6 选取meantemp作为单变量时间序列
        X.append(
            data[:, feature_index][i : i + lookback]
        )  # [i : i + lookback] 左闭右开区间为特征值
        y.append(
            data[:, feature_index][i + lookback : i + lookback + forcast_lengths]
        )  # 选择[i + lookback : i + lookback + forcast_lengths]区间为目标值
        # print(np.array(X).shape)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return torch.tensor(X), torch.tensor(y)
