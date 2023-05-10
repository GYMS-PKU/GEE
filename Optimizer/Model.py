# Copyright (c) 2022 Dai HBG

"""
为GEE定义常见的模型，其中需要指定激活函数

2022-12-17
- 新增probit
2022-09-26
- 取消Linear的bias
"""


import torch


class GEEModel(torch.nn.Module):
    def __init__(self, input_dim: int, link_func: str = 'sigmoid'):
        super(GEEModel, self).__init__()
        self.Dense = torch.nn.Linear(input_dim, 1, bias=False)
        if link_func == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif link_func == 'probit':
            self.act = torch.distributions.normal.Normal(0, 1)
        else:
            raise NotImplementedError
        self.link_func = link_func
        self.input_dim = input_dim

    def forward(self, x):
        if self.link_func == 'sigmoid':
            return self.act(self.Dense(x))
        elif self.link_func == 'probit':
            return self.act.cdf(self.Dense(x))
        else:
            raise NotImplementedError

