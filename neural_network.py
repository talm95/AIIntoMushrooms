import torch
from torch import nn
from data_prepare import get_categories

categories = get_categories()


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_ = nn.Linear(101, 64)
        self.hidden1 = nn.Linear(64, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 9)

    def forward(self, x):
        x = self.input_(x)
        x = torch.relu(x)
        x = self.hidden1(x)
        x = torch.tanh(x)
        x = self.hidden2(x)
        x = torch.tanh(x)
        x = self.output(x)
        return x

