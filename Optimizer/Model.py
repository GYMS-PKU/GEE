# Copyright (c) 2022 Dai HBG

"""
为GEE定义常见的模型，其中需要指定激活函数
"""


import torch


class GEEModel(torch.nn.Module):
    def __init__(self, input_dim: int, link_func: str = 'sigmoid'):
        super(GEEModel, self).__init__()
        self.Dense = torch.nn.Linear(input_dim, 1)
        if link_func == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            raise NotImplementedError
        self.input_dim = input_dim

    def forward(self, x):
        return self.act(self.Dense(x))

