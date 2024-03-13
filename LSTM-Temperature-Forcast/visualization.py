# Plot
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
from plotly.graph_objects import Box, Scatter


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
        fig.write_image("LSTM-Temperature-Forcast/fig/box_plot.png")

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
        fig.write_image("LSTM-Temperature-Forcast/fig/line_plot.png")
