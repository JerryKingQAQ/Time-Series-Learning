# Plot
from os import mkdir
import os
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
from plotly.graph_objects import Box, Scatter
import plotly.graph_objects as go


class plotly_graph:
    def __init__(self, data, time):
        self.data = data
        self.time = time
        self.name_list = ["Mean Temp", "Humidity", "Wind Speed", "Mean Pressure"]
        self.box_title = "Multiple Box Plots"
        self.line_title = "Multiple Line Plots"

    def make_subplot(self, graphs):  # 构建子图
        fig = make_subplots(rows=2, cols=2, subplot_titles=self.name_list)
        for i in range(len(self.name_list)):
            fig.add_trace(graphs[i], row=i // 2 + 1, col=i % 2 + 1)
        return fig

    def box_plot(self):  # 构建箱型图
        graph_list = []
        for i, element in enumerate(self.data.transpose()):
            graph = Box(
                y=element,
                name=self.name_list[i],
                boxpoints="outliers",
                line=dict(width=1),
            )
            graph_list.append(graph)
        fig = self.make_subplot(graph_list)
        fig.update_layout(
            title=self.line_title,
            xaxis_title="Time",
            yaxis_title="Value",
            template="simple_white",
            showlegend=False,
        )
        fig.write_image(
            "/home/jinjiarui/run/Time-Series-Learning/LSTM-Temperature-Forecast/fig/box_plot.png"
        )

    def line_plot(self):  # 构建折线图
        graph_list = []
        for i, element in enumerate(self.data.transpose()):
            graph = Scatter(
                x=self.time, y=element, name=self.name_list[i], mode="lines"
            )
            graph_list.append(graph)
        fig = self.make_subplot(graph_list)
        fig.update_layout(
            title=self.line_title,
            xaxis_title="Time",
            yaxis_title="Value",
            template="simple_white",
            showlegend=False,
        )
        fig.write_image(
            "/home/jinjiarui/run/Time-Series-Learning/LSTM-Temperature-Forecast/fig/line_plot.png"
        )

    def draw_pred_pic(self, train_preds, test_preds, feature_index=0):
        # 分别为训练集和测试集创建时间序列
        time_train = self.time[:len(train_preds)]
        time_test = self.time[-len(test_preds):]

        # 计算在时间序列中开始绘制预测值的索引
        start_train_idx = len(self.time) - len(train_preds)
        start_test_idx = len(self.time) - len(test_preds)

        # 选取要展示的特征列，这里假设是平均温度，即第0列
        real_train = self.data[start_train_idx:, feature_index]
        real_test = self.data[start_test_idx:, feature_index]

        # 创建子图，每个数据集一个图
        fig = make_subplots(rows=2, cols=1, subplot_titles=["Training Set", "Test Set"])

        # 添加训练集的真实值与预测值
        fig.add_trace(
            go.Scatter(
                x=time_train,
                y=real_train,
                mode="lines",
                name="Train Real",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_train,
                y=train_preds,
                mode="lines",
                name="Train Predicted",
                line=dict(color="red"),
            ),
            row=1,
            col=1,
        )

        # 添加测试集的真实值与预测值
        fig.add_trace(
            go.Scatter(
                x=time_test,
                y=real_test,
                mode="lines",
                name="Test Real",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_test,
                y=test_preds,
                mode="lines",
                name="Test Predicted",
                line=dict(color="orange"),
            ),
            row=2,
            col=1,
        )

        # 更新布局
        fig.update_layout(
            title="Real vs Predicted Values",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=True,
        )
        # 保存图像
        fig.write_image(
            "/home/jinjiarui/run/Time-Series-Learning/LSTM-Temperature-Forecast/fig/prediction_comparison.png"
        )
