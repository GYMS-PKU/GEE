# Copyright (c) 2022-2023 Dai HBG


"""
一个solver的框架
需要整体初始化的有K个cluster，每个cluster里有对应的X，Y，alpha
还需要一个额外的记录alpha位置的变量，在更新alpha时需要用到

2023-04-29
- add a simple optimization mode, using l2 loss to optimize the beta only
2023-04-15
- loss在更新时需要定时重新计算梯度
2023-04-11
- 新增一个L2 Loss，比较收敛速度
2023-03-28
mark：
在估计多alpha的时候，传入的alpha是可加的，所以需要额外计算重叠的部分，然后解出需要的alpha
2022-12-17
- 新增probit
2022-09-21
- 修正bug：多alpha时，并不能划分成等价类，因此alpha的index必须指定使用这个alpha的所有共同对
2022-09-08
mark：
有两个地方需要计算alpha的信息，一个是计算协方差矩阵时，每个cluster有一个矩阵；
第二个地方是更新alpha时，需要获得所有共用alpha的位置

针对这个情况，初始化模型的时候需要传入alpha的位置和值字典，形式是：
alpha_i: {'index': [mat_1, ..., mat_K], 'value': value}
其中mat_i是0-1矩阵，第i行j列为1表示i和j使用该相关系数，同时满足是对称的，因为要更新alpha矩阵
K是cluster的个数

2022-09-03
- init
"""

import torch

import sys

sys.path.append('C:/Users/Administrator/Desktop/Repositories/GEE')

from Optimizer.func_dic import link_func_dic, var_func_dic  # 连接函数字典
from Optimizer.Model import GEEModel

from tqdm import tqdm


class LossMaker:
    def __init__(self, link_func: str = 'sigmoid', var_func: str = 'binary', device: str = 'cuda'):
        """
        :param link_func: 均值链接函数
        :param var_func: 方差和均值的函数
        """
        self.mean_link = link_func_dic[link_func]
        self.var_func = var_func_dic[var_func]
        self.device = device

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

            s_k_inv = torch.eye(len(x)).to(self.device) * 1 / torch.sqrt(self.var_func(model(x)) * phi)
            alpha = clusters[k]['alpha']  # 相关性矩阵
            alpha_inv = torch.inverse(alpha)  # 相关系数矩阵的逆

            v_k = torch.matmul(torch.matmul(s_k_inv, alpha_inv), s_k_inv)

            # print(torch.max(torch.abs(v_k)).item)
            v_k = v_k.detach()  # 去掉梯度
            v_inv.append(v_k)

        return d_top, v_inv

    def make_loss(self, phi, clusters, model, loss: str = 'l1'):  # 获得损失函数
        d_top, v_inv = self.cal_mat(phi, clusters, model)
        if loss == 'l1':
            return Loss_l1(d_top, v_inv)
        else:
            return Loss_l2(d_top, v_inv)


class Loss_l1(torch.nn.Module):  # 求解估计方程的损失函数
    def __init__(self, d_top, v_inv):
        super(Loss_l1, self).__init__()
        self.d_top = d_top
        self.v_inv = v_inv

    def forward(self, y: list, mu: list):
        # output的维度是input_dim * 1
        k = 0
        output = torch.matmul(torch.matmul(self.d_top[k], self.v_inv[k]), y[k] - mu[k])
        for k in range(1, len(y)):
            output += torch.matmul(torch.matmul(self.d_top[k], self.v_inv[k]), y[k] - mu[k])
        return torch.mean(torch.abs(output))


class Simple_Loss(torch.nn.Module):  # 求解估计方程的损失函数
    def __init__(self):
        super().__init__()

    def forward(self, y: list, mu: list):
        # output的维度是input_dim * 1
        loss = 0
        count = 0
        for i in range(len(y)):
            loss += torch.sum((y[i] - mu[i]) ** 2)
            count += len(y[i])
        return loss / count


class Loss_l2(torch.nn.Module):  # 求解估计方程的损失函数
    def __init__(self, d_top, v_inv):
        super(Loss_l2, self).__init__()
        self.d_top = d_top
        self.v_inv = v_inv

    def forward(self, y: list, mu: list):
        # output的维度是input_dim * 1
        k = 0
        output = torch.matmul(torch.matmul(self.d_top[k], self.v_inv[k]), y[k] - mu[k])
        for k in range(1, len(y)):
            output += torch.matmul(torch.matmul(self.d_top[k], self.v_inv[k]), y[k] - mu[k])
        return torch.mean(output ** 2)


class ModelLogit(torch.nn.Module):  # 使用Logit作为激活函数的模型
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.Dense = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.Dense(x)
        x = 1 / (1 + torch.exp(x))
        return x


class GEE:
    def __init__(self, alphas: dict, clusters: list, X: torch.Tensor,
                 link_func: str = 'sigmoid', var_func: str = 'binary', device: str = 'cuda', loss_name: str = 'l1'):
        """
        :param alphas: alpha字典
        :param clusters: cluster列表
        :param link_func: 均值的连接函数
        :param device:
        """
        self.alphas = alphas
        self.clusters = clusters
        u, s, v = torch.linalg.svd(X, full_matrices=False)
        self.H = torch.matmul(torch.matmul(v.T, torch.diag(1 / s)), u.T)  # 投影矩阵
        self.model = GEEModel(input_dim=clusters[0]['x'].shape[1], link_func=link_func)  # 生成一个模型
        self.device = device
        self.model.to(self.device)

        self.loss_maker = LossMaker(link_func=link_func, var_func=var_func, device=device)
        self.phi = 1  # 初始化phi
        self.loss = self.loss_maker.make_loss(self.phi, clusters=clusters, model=self.model,
                                              loss=loss_name)  # 初始化损失函数
        self.simple_loss = Simple_Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.loss_name = loss_name

    def fit(self, epochs: int = 10, n_iter: int = 100, grad_renew_fre: int = 1000, verbose: bool = False):
        """
        :param epochs: 交替优化的轮数
        :param n_iter: 迭代次数
        :param grad_renew_fre: 梯度更新频次
        :param verbose: 是否打印loss
        """
        self.solve(epochs=epochs, n_iter=n_iter, grad_renew_fre=grad_renew_fre, verbose=verbose)

    def simple_fit(self, epochs: int = 10, n_iter: int = 100, grad_renew_fre: int = 1000, verbose: bool = False):
        """
        :param epochs: 交替优化的轮数
        :param n_iter: 迭代次数
        :param grad_renew_fre: 梯度更新频次
        :param verbose: 是否打印loss
        """
        self.simple_solve(epochs=epochs, n_iter=n_iter, grad_renew_fre=grad_renew_fre, verbose=verbose)

    def solve(self, epochs: int = 10, n_iter: int = 100, grad_renew_fre: int = 100,
              verbose: bool = False):
        """
        :param epochs: 交替优化的轮数
        :param n_iter: 迭代次数
        :param grad_renew_fre: 梯度更新频次
        :param verbose: 是否打印loss
        """
        for n in range(epochs):
            self.optimize(n_iter, grad_renew_fre, verbose=verbose)
            self.renew_param()  # 更新alpha和phi
            self.loss = self.loss_maker.make_loss(self.phi, clusters=self.clusters, model=self.model,
                                                  loss=self.loss_name)  # 更新loss

    def simple_solve(self, epochs: int = 10, n_iter: int = 100, grad_renew_fre: int = 100,
                     verbose: bool = False):
        """
        :param epochs: 交替优化的轮数
        :param n_iter: 迭代次数
        :param grad_renew_fre: 梯度更新频次
        :param verbose: 是否打印loss
        """
        for _ in range(epochs):
            self.simple_optimize(n_iter, grad_renew_fre, verbose=verbose)
            self.renew_param()  # 更新alpha和phi

    def optimize(self, n_iter: int = 100, grad_renew_fre: int = 1000, verbose: bool = False):  # 固定loss优化beta
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

            if (i + 1) % grad_renew_fre == 0:  # 更新梯度信息
                self.loss = self.loss_maker.make_loss(self.phi, clusters=self.clusters, model=self.model,
                                                      loss=self.loss_name)  # 更新loss

            if verbose:
                if (i + 1) % (n_iter // 10) == 0:
                    print('loss: {:.4f}'.format(loss.item()))

        print('loss: {:.4f}'.format(loss.item()))

    def simple_optimize(self, n_iter: int = 100, grad_renew_fre: int = 1000,
                        verbose: bool = False):  # optimize beta
        for i in range(n_iter):
            y = []
            mu = []
            for k in range(len(self.clusters)):
                y.append(self.clusters[k]['y'])
                mu.append(self.model(self.clusters[k]['x']))
            loss = self.simple_loss(y, mu)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if (i + 1) % grad_renew_fre == 0:  # 更新梯度信息
            #     self.loss = self.loss_maker.make_loss(self.phi, clusters=self.clusters, model=self.model,
            #                                           loss=self.loss_name)  # 更新loss

            if verbose:
                if (i + 1) % (n_iter // 10) == 0:
                    print('loss: {:.4f}'.format(loss.item()))

        print('loss: {:.4f}'.format(loss.item()))

    def renew_param(self):  # 更新alpha和phi
        # 更新phi
        phi = 0
        N = 0

        res = []  # 保存预测均值，长度是cluster的个数
        var = []  # 保存方差，长度是cluster的个数

        Y = []  # 用于求解alpha的Y

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
        # 需要使用最小二乘来同时估计全部的alpha
        for kk in range(len(res)):  # 然后对每个cluster循环
            r = res[kk] / torch.sqrt(self.phi * var[kk])  # 标准化残差，形状是n_k * 1
            r = torch.matmul(r, r.T)  # 生成所有的交错乘积
            Y.append(r.reshape(-1, 1))  # 这是对应的r_{k,i,j}
        Y = torch.concat(Y, dim=0)
        alpha = torch.matmul(self.H, Y)[:, 0]  # 依次的alpha
        for k in self.alphas.keys():
            self.alphas[k]['value'] = float(alpha[int(k)])  # 更新alpha

        # 更新相关系数矩阵
        # 注意因为相关系数可加性，需要从I开始重新计算
        for kk in range(len(self.clusters)):
            self.clusters[kk]['alpha'] = torch.eye(len(self.clusters[kk]['alpha'])).to(self.device)  # 初始化
        for k, v in self.alphas.items():
            index = v['index']
            value = v['value']
            for kk in range(len(self.clusters)):
                self.clusters[kk]['alpha'][index[kk]] += value  # 更新值
