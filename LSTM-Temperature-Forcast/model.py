import torch
import torch.nn as nn


class BiLSTM(nn.Modul):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=6,
            num_layers=2,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.linear2 = nn.Linear(64, 8)
        self.out_linear = nn.Linear(8, 1)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.out_linear(x)
        return x
