# Copyright (c) 2022 Dai HBG

"""
为GEE定义常见的方差计算函数
"""


import torch


def probit_var(x: torch.Tensor):
    return x * (1 - x)


def sigmoid_link(x: torch.Tensor):
    return 1 / (1 + torch.exp(x))


link_func_dic = {'sigmoid': sigmoid_link}  # 连接函数字典，可以直接调用
var_func_dic = {'probit': probit_var}  # 方差函数字典，直接调用
