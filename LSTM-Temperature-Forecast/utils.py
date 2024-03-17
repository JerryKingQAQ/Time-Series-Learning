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


def create_time_dataset(
    data, lookback, forecast_lengths, device, step=1
):
    X, y = [], []
    for i in range(0, len(data) - lookback - forecast_lengths + 1, step):
        X.append(
            data[i : i + lookback]
        )  # [i : i + lookback] 左闭右开区间为特征值
        y.append(
            data[i + lookback : i + lookback + forecast_lengths]
        )  # 选择[i + lookback : i + lookback + forecast_lengths]区间为目标值
        # print(np.array(X).shape)

    X = np.array(X, dtype=np.float32).squeeze()
    y = np.array(y, dtype=np.float32).squeeze()
    return torch.tensor(X).to(device), torch.tensor(y).to(device)


import numpy as np


def average_predictions(predictions, total_length, forecast_length, lookback):
    total_predictions = np.zeros(total_length)
    count_predictions = np.zeros(total_length)

    offset = lookback  # 从lookback值开始计算
    for batch in predictions:
        for pred in batch:
            pred = pred.astype(np.float64)

            end = min(offset + forecast_length, total_length)
            total_predictions[offset:end] += pred[: end - offset]
            count_predictions[offset:end] += 1
            offset += 1
            if offset >= total_length:
                break
        if offset >= total_length:
            break

    # 计算平均值但避免在lookback区间内计算
    average_predictions = np.zeros(total_length)  # 初始化平均值数组
    valid_counts = count_predictions != 0  # 找出非零计数的索引
    average_predictions[valid_counts] = (
        total_predictions[valid_counts] / count_predictions[valid_counts]
    )

    # lookback之前的值已经默认是0，因此无需进一步操作

    return average_predictions
