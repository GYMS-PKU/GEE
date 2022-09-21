# Copyright (c) 2022 Dai HBG


"""
一个solver的框架
需要整体初始化的有K个cluster，每个cluster里有对应的X，Y，alpha
还需要一个额外的记录alpha位置的变量，在更新alpha时需要用到

2022-09-08
mark：
有两个地方需要计算alpha的信息，一个是计算协方差矩阵时，每个cluster有一个矩阵；
第二个地方是更新alpha时，需要获得所有共用alpha的位置

针对这个情况，初始化模型的时候需要传入alpha的位置和值字典，形式是：
alpha_i: {'index': [mat_1, ..., mat_k], 'value': value}
其中mat_i是0-1矩阵，第i行j列为1表示i和j使用该相关系数，同时满足是对称的，因为要更新alpha矩阵

2022-09-21
- 修正bug：多alpha时，并不能划分成等价类，因此alpha的index必须指定使用这个alpha的所有共同对
2022-09-03
- init
"""

import torch

import sys

sys.path.append('C:/Users/Administrator/Desktop/Repositories/GEE')

from Optimizer.func_dic import link_func_dic, var_func_dic  # 连接函数字典
from Optimizer.Model import GEEModel


class LossMaker:
    def __init__(self, link_func: str = 'probit', var_func: str = 'binary'):
        """
        :param link_func: 均值链接函数
        :param var_func: 方差和均值的函数
        """
        self.mean_link = link_func_dic[link_func]
        self.var_func = var_func_dic[var_func]

    def cal_mat(self, phi: float, clusters, model):  # 返回GEE1.0中所需的矩阵
        """
        :param phi: 当前的phi系数
        :param clusters:
        :param model:
        :return:
        """
        d_top = []
        v_inv = []
        # 计算D_k和V_k的逆
        for k in range(len(clusters)):  # 对K个cluster循环
            d_k = []
            x = clusters[k]['x']  # 一个N_k * input_dim的矩阵
            for j in range(len(x)):  # 对每个样本计算梯度
                model.zero_grad()
                y = model(x[j])
                y.backward()
                d_k.append(torch.clone(model.Dense.weight.grad))
            d_k = torch.cat(d_k, dim=0)  # 维度是N_k * input_dim
            d_k = d_k.detach()  # 去掉梯度
            d_top.append(d_k.T)

            s_k_inv = torch.eye(len(x)).cuda() * 1 / torch.sqrt(self.var_func(model(x)) * phi)
            alpha = clusters[k]['alpha']  # 相关性矩阵
            alpha_inv = torch.inverse(alpha)  # 相关系数矩阵的逆
            v_k = torch.matmul(torch.matmul(s_k_inv, alpha_inv), s_k_inv)
            v_k = v_k.detach()  # 去掉梯度
            v_inv.append(v_k)

        return d_top, v_inv

    def make_loss(self, phi, clusters, model):  # 获得损失函数
        d_top, v_inv = self.cal_mat(phi, clusters, model)
        return Loss(d_top, v_inv)


class Loss(torch.nn.Module):  # 求解估计方程的损失函数
    def __init__(self, d_top, v_inv):
        super(Loss, self).__init__()
        self.d_top = d_top
        self.v_inv = v_inv

    def forward(self, y: list, mu: list):
        k = 0
        # output的维度是input_dim * 1
        output = torch.matmul(torch.matmul(self.d_top[k], self.v_inv[k]), y[k] - mu[k])
        for k in range(1, len(y)):
            output += torch.matmul(torch.matmul(self.d_top[k], self.v_inv[k]), y[k] - mu[k])
        return torch.mean(torch.abs(output))


class ModelLogit(torch.nn.Module):  # 使用Logit作为激活函数的模型
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.Dense = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.Dense(x)
        x = 1 / (1 + torch.exp(x))
        return x


class GEE:  # 初始化需要传入
    def __init__(self, alphas: dict, clusters: list, link_func: str = 'sigmoid', var_func: str = 'probit',
                 device: str = 'cuda'):
        """
        :param alphas: alpha字典
        :param clusters: cluster列表
        :param link_func: 均值的连接函数
        :param device:
        """
        self.alphas = alphas
        self.clusters = clusters
        self.model = GEEModel(input_dim=clusters[0]['x'].shape[1], link_func=link_func)  # 生成一个模型
        if device == 'cuda':
            self.model.cuda()

        self.loss_maker = LossMaker(link_func=link_func, var_func=var_func)
        self.phi = 1  # 初始化phi
        self.loss = self.loss_maker.make_loss(self.phi, clusters=clusters, model=self.model)  # 初始化损失函数
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def fit(self, epochs: int = 10, n_iter: int = 100):
        """
        :param epochs: 交替优化的轮数
        :param n_iter: 迭代次数
        """
        self.solve(epochs=epochs, n_iter=n_iter)

    def solve(self, epochs: int = 10, n_iter: int = 100):
        """
        :param epochs: 交替优化的轮数
        :param n_iter: 迭代次数
        """
        for n in range(epochs):
            self.optimize(n_iter)
            self.renew_param()  # 更新alpha和phi
            self.loss = self.loss_maker.make_loss(self.phi, clusters=self.clusters, model=self.model)  # 更新loss

    def optimize(self, n_iter: int = 100):  # 固定loss优化beta
        for i in range(n_iter):
            y = []
            mu = []
            for k in range(len(self.clusters)):
                y.append(self.clusters[k]['y'])
                mu.append(self.model(self.clusters[k]['x']))
            loss = self.loss(y, mu)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(loss.item())

    def renew_param(self):  # 更新alpha和phi
        # 更新phi
        phi = 0
        N = 0

        res = []  # 保存预测均值，长度是cluster的个数
        var = []  # 保存方差，长度是cluster的个数

        for k in range(len(self.clusters)):
            mu = self.model(self.clusters[k]['x'])
            v = self.loss_maker.var_func(mu)
            r = self.clusters[k]['y'] - mu
            res.append(r)
            var.append(v)
            phi += torch.sum(r ** 2 / v).item()
            N += len(mu)
        phi /= (N - self.model.input_dim)
        self.phi = phi

        # 更新alpha
        for k, v in self.alphas.items():
            index = v['index']
            count = 0
            alpha = 0
            for kk in range(len(res)):
                r = res[kk] / torch.sqrt(self.phi * var[kk])  # 标准化残差，形状是n_k * 1
                r = torch.matmul(r, r.T)  # 生成所有的交错乘积
                count += torch.sum(index[kk]).item() / 2  # 这就是交错乘积的个数
                alpha += torch.sum(r * index[kk]).item() / 2  # 残差的累积交错乘积
            self.alphas[k]['value'] = alpha / (count - self.model.input_dim)  # 更新alpha

        # 更新相关系数矩阵
        for k, v in self.alphas.items():
            index = v['index']
            value = v['value']
            for kk in range(len(self.clusters)):
                self.clusters[kk]['alpha'][index[kk]] = value  # 更新值
