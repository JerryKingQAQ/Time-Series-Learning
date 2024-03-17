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
        self.linear2 = nn.Linear(64, 32)
        self.out_linear = nn.Linear(32, self.output_size)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.out_linear(x)
        return x
