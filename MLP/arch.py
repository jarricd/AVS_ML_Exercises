import os
import torch


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 28*28),
            torch.nn.ReLU(),
            torch.nn.Linear(28*28, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


class MLP10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 28*28),
            torch.nn.ReLU(),
            torch.nn.Linear(28*28, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


class MLP20(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(20, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 28*28),
            torch.nn.ReLU(),
            torch.nn.Linear(28*28, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)


class MLP30(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(30, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 28*28),
            torch.nn.ReLU(),
            torch.nn.Linear(28*28, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)

        )

    def forward(self, x):
        return self.layers(x)