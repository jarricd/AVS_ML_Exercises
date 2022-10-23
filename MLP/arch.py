import os
import torch


class MLP(torch.nn.Module):
    """
    REMEMBER:
    For matrix multiplication,
    the number of columns in the first matrix must
    be equal to the number of rows in the second matrix.
    The result matrix has the number of
    rows of the first and the number of columns of the second matrix.
    """
    def __init__(self):
        super().__init__()
        # this code is written for LDA-reduced dataset to 2 dimensions
        # the model it's hopeless
        # since we have input 10 features, we have to 'transplant' it into more output features
        # we need more dimensions and modified architectures
        self.layers = torch.nn.Sequential(
            # torch.nn.Flatten(), # might be needed to pool everything into one layer
            # apply following function: y = x * A^T + b, where y is out, A - weights, b is bias
            # if an input is [60000, 2], then first param is num features, output is num features we get
            torch.nn.Linear(2, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 10),
            torch.nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)


class MLP10(torch.nn.Module):
    """
    REMEMBER:
    For matrix multiplication,
    the number of columns in the first matrix must
    be equal to the number of rows in the second matrix.
    The result matrix has the number of
    rows of the first and the number of columns of the second matrix.
    """
    def __init__(self):
        super().__init__()
        # this code is written for LDA-reduced dataset to 2 dimensions
        # the model it's hopeless
        # since we have input 10 features, we have to 'transplant' it into more output features
        # we need more dimensions and modified architectures
        self.layers = torch.nn.Sequential(
            # torch.nn.Flatten(), # might be needed to pool everything into one layer
            # apply following function: y = x * A^T + b, where y is out, A - weights, b is bias
            # if an input is [60000, 2], then first param is num features, output is num features we get
            torch.nn.Linear(10, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 10),
            torch.nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)


class MLP20(torch.nn.Module):
    """
    REMEMBER:
    For matrix multiplication,
    the number of columns in the first matrix must
    be equal to the number of rows in the second matrix.
    The result matrix has the number of
    rows of the first and the number of columns of the second matrix.
    """
    def __init__(self):
        super().__init__()
        # this code is written for LDA-reduced dataset to 2 dimensions
        # the model it's hopeless
        # since we have input 10 features, we have to 'transplant' it into more output features
        # we need more dimensions and modified architectures
        self.layers = torch.nn.Sequential(
            # torch.nn.Flatten(), # might be needed to pool everything into one layer
            # apply following function: y = x * A^T + b, where y is out, A - weights, b is bias
            # if an input is [60000, 2], then first param is num features, output is num features we get
            torch.nn.Linear(20, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 40),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 10),
            torch.nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)


class MLP30(torch.nn.Module):
    """
    REMEMBER:
    For matrix multiplication,
    the number of columns in the first matrix must
    be equal to the number of rows in the second matrix.
    The result matrix has the number of
    rows of the first and the number of columns of the second matrix.
    """
    def __init__(self):
        super().__init__()
        # this code is written for LDA-reduced dataset to 2 dimensions
        # the model it's hopeless
        # since we have input 10 features, we have to 'transplant' it into more output features
        # we need more dimensions and modified architectures
        self.layers = torch.nn.Sequential(
            # torch.nn.Flatten(), # might be needed to pool everything into one layer
            # apply following function: y = x * A^T + b, where y is out, A - weights, b is bias
            # if an input is [60000, 2], then first param is num features, output is num features we get
            torch.nn.Linear(30, 90),
            torch.nn.ReLU(),
            torch.nn.Linear(90, 45),
            torch.nn.ReLU(),
            torch.nn.Linear(45, 10),

        )

    def forward(self, x):
        return self.layers(x)