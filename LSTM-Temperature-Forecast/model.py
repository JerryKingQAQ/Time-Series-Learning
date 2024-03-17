import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.bilstm = nn.LSTM(
            input_size=self.input_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(self.hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 32)
        self.out_linear = nn.Linear(32, self.output_size)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.out_linear(x)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(self.hidden_size * 1, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 32)
        self.out_linear = nn.Linear(32, self.output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.out_linear(x)
        return x


class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size ,output_size):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))
