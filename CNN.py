import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential  (
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 11),

        )
    def forward(self, x):
        return self.network(x)

