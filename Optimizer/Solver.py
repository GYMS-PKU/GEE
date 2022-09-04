# Copyright (c) 2022 Dai HBG


"""
一个solver的框架
需要整体初始化的有K个cluster，每个cluster里有对应的X，Y，alpha
还需要一个额外的记录alpha位置的变量，在更新alpha时需要用到

2022-09-03
- init
"""


import torch


link_func_dic = {}  # 连接函数字典，可以直接调用
var_func_dic = {}  # 方差函数字典，直接调用


class LossMaker:
    def __init__(self, mean_link: str = 'probit', variance_func: str = 'binary',
                 clusters=None, model=None):
        """
        :param mean_link: 均值链接函数
        :param variance_func: 方差和均值的函数
        :param clusters: 初始化的cluster
        :param model: 模型，对应的是mu = g^{-1}(X^\top\beta)
        """
        self.mean_link = link_func_dic[mean_link]
        self.variance_func = var_func_dic[variance_func]
        self.clusters = clusters
        self.model = model

    def cal_mat(self, phi: float):  # 返回GEE1.0中所需的矩阵
        """
        :param phi: 当前的phi系数
        :return:
        """
        d_top = []
        v_inv = []
        # 计算D_k和V_k的逆
        for k in range(len(self.clusters)):  # 对K个cluster循环
            d_k = []
            x = self.clusters[k]['x']  # 一个N_k * input_dim的矩阵
            for j in range(len(x)):  # 对每个样本计算梯度
                self.model.zero_grad()
                y = self.model()
                y.backward()
                d_k.append(torch.clone(self.model.Dense.weight.grad))
            d_k = torch.cat(d_k, dim=0)  # 维度是N_k * input_dim
            d_top.append(d_k.T)

            s_k_inv = torch.eye(len(x)) * 1 / torch.sqrt(self.variance_func(self.model(x)) * phi)
            alpha = self.clusters[k]['alpha']  # 相关性矩阵
            alpha_inv = torch.inverse(alpha)  # 相关系数矩阵的逆
            v_k = torch.matmul(torch.matmul(s_k_inv, alpha_inv), s_k_inv)
            v_inv.append(v_k)

        return d_top, v_inv

    def make_loss(self, phi):  # 获得损失函数
        d_top, v_inv = self.cal_mat(phi)
        return Loss(d_top, v_inv)


class Loss(torch.nn.Module):
    def __init__(self, d_top, v_inv):
        super().__init__()
        self.d_top = d_top
        self.v_inv = v_inv

    def forward(self, y: list, mu: list):
        k = 0
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


class Solver:
    def __init__(self, model, clusters, alphas: dict, loss_maker: LossMaker):
        """
        :param model: 模型
        :param clusters: 族
        :param alphas: 记录alpha结构的字典，结构是i: [index]，形状和clusters一致，
        :param loss_maker: 损失函数构造类
        """
        self.model = model
        self.clusters = clusters
        self.alphas = alphas
        self.loss_maker = loss_maker
        self.phi = 1  # 初始化phi
        self.loss = loss_maker.make_loss(self.phi)  # 初始化损失函数
        self.optimizer = torch.optim.AdamW(self.model, lr=1e-3)

    def solve(self, epochs: int = 10, n_iter: int = 100):
        """
        :param epochs: 交替优化的轮数
        :param n_iter: 迭代次数
        """
        for n in range(epochs):
            self.optimize(n_iter)
            self.renew_param()  # 更新alpha和phi
            self.loss = self.loss_maker.make_loss(self.phi)  # 更新loss

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

    def renew_param(self):  # 更新alpha和phi
        # 更新phi
        phi = 0
        N = 0
        for k in range(len(self.clusters)):
            mu = self.model(self.clusters[k]['x'])
            v = self.loss_maker.variance_func(mu)
            phi += torch.sum((self.clusters[k]['y'] - mu) / v).item()
            N += len(mu)
        phi /= (N - self.model.input_dim)
        self.phi = phi

        # 更新alpha
        for i in range(len(self.alphas)):
            pass

