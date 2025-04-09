import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential  (nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Linear(in_features=54*3, out_features=1)

            )
    def forward(self, x):
        return self.network(x)

