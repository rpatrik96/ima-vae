from typing import Literal

from torch import nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


ActivationType = Literal["lrelu", "sigmoid", "none"]
PriorType = Literal["gaussian", "beta", "uniform", "laplace"]
