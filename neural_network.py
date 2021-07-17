import torch
from torch import nn
from data_prepare import get_categories

categories = get_categories()


class Network(nn.Module):
    def __init__(self, number_of_features):
        super().__init__()

        self.input_ = nn.Linear(number_of_features, 100)
        self.output = nn.Linear(100, 9)

    def forward(self, x):
        x = self.input_(x)
        x = torch.relu(x)
        x = self.output(x)
        return x

