import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = nn.Sequential  (
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(27 * 27,200),
            nn.ReLU(True),
            nn.Linear(200,11),
            nn.LogSoftmax(dim=1)
        )



    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


