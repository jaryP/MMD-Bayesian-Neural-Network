from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from base import Network, Gaussian, Wrapper, get_bayesian_network
from bayesian_utils import BayesianCNNLayer, BayesianLinearLayer


# def euclidean_dist( x, y):
#     # x: N x D
#     # y: M x D
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     assert d == y.size(1)
#
#     # x = x.unsqueeze(1).expand(n, m, d)
#     # y = y.unsqueeze(0).expand(n, m, d)
#
#     return torch.pow(torch.sub(x.unsqueeze(1).expand(n, m, d), y.unsqueeze(0).expand(n, m, d)), 2).sum(2)
#
#
# def pairwise_distances(x, y):
#
#     x_norm = (x**2).sum(1).view(-1, 1)
#     y_norm = (y ** 2).sum(1).view(1, -1)
#
#     dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
#     return torch.clamp(dist, 0.0, np.inf)
#
#
# def cdist2(x, y):
#     # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
#     x_sq_norm = x.pow(2).sum(dim=-1)
#     y_sq_norm = y.pow(2).sum(dim=-1)
#     x_dot_y = x @ y.t()
#     sq_dist = x_sq_norm.unsqueeze(dim=1) + y_sq_norm.unsqueeze(dim=0) - 2*x_dot_y
#     # For numerical issues
#     sq_dist.clamp_(min=0.0)
#     return torch.sqrt(sq_dist)
#
#
# def compute_kernel(x, y):
#     dim = x.size(1)
#     # d = pairwise_distances(x, y)
#     # print(d.size())
#     # d = torch.exp(- torch.mul(torch.cdist(x, y).mean(1), 1/float(dim))).mean()
#     d = torch.exp(- torch.mul(pairwise_distances(x, y).mean(1), 1/float(dim))).mean()
#     # d = torch.exp(- torch.mul(pairwise_distances1(x, y).mean(1), 1/float(dim))).mean()
#
#     # d2 = cdist2(x, y)
#     # d1 = torch.norm(x[:, None] - y, dim=2, p=2)
#     # d = (- torch.norm(x[:, None] - y, dim=2, p=2).mean(1) / float(dim)).exp().mean()
#     # d = (- row_pairwise_distances(x, y) / float(dim)).exp().mean().to(x.device)
#     # d1 = x**2 + y**2 - 2*torch.matmul(x.t(), y)
#     # d1 = torch.norm(x - y, 2)
#     return d
#
#
# def compute_mmd(x, y):
#     x_kernel = compute_kernel(x, x)
#     y_kernel = compute_kernel(y, y)
#     xy_kernel = compute_kernel(x, y)
#     return x_kernel + y_kernel - 2 * xy_kernel


class BMMD(Network):

    def __init__(self, sample, classes, topology=None, prior=None, mu_init=None, rho_init=None,
                 local_trick=False, regression=False, posterior_type='weights', **kwargs):
        super().__init__(classes, regression)

        if topology is None:
            topology = [400, 400]

        if prior is None:
            prior = Gaussian(0, 10)

        self.calculate_mmd = True
        self._prior = prior
        self.features = get_bayesian_network(topology, sample, classes,
                                             mu_init, rho_init, prior, 'mmd', local_trick, posterior_type, bias=True,
                                             **kwargs)

        # ##################### non abbandonarmi più jary!!!!
        # ##################### Scusa :'(

    def _forward(self, x):

        mmd = 0
        for j, i in enumerate(self.features):
            if not isinstance(i, (BayesianLinearLayer, BayesianCNNLayer)):
                x = i(x)
            else:
                x, m = i(x, self.calculate_mmd)
                mmd += m

        return x, mmd

    def forward(self, x, samples=1):
        o = []
        mmds = []

        for i in range(samples):
            op, mmd = self._forward(x)
            o.append(op)
            mmds.append(mmd)

        o = torch.stack(o)

        mmds = torch.stack(mmds).mean()

        return o, mmds

    def eval_forward(self, x, samples=1):
        o, _ = self(x, samples=samples)
        return o


class Trainer(Wrapper):
    def __init__(self, model: nn.Module, train_data, test_data, optimizer):
        super().__init__(model, train_data, test_data, optimizer)

    def train_epoch(self, samples=1, **kwargs):
        losses = []

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=False, leave=False)
        progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

        mmd_w = kwargs.get('weights', {}).get('mmd', 1)

        train_true = []
        train_pred = []

        M = len(self.train_data)
        a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
        b = 2 ** M - 1

        pi = a / b
        self.model.calculate_mmd = True

        for batch, (x, y) in progress_bar:

            train_true.extend(y.tolist())

            y = y.to(self.device)
            x = x.to(self.device)
            self.optimizer.zero_grad()

            out, mmd = self.model(x, samples=samples)
            # out = out.mean(0)

            mmd *= mmd_w
            # mmd /= x.shape[0]
            mmd *= pi[batch]
            # mmd *= 1/M

            if pi[batch] == 0:
                self.model.calculate_mmd = False

            if self.regression:
                out = out.mean(0)
                if self.model.classes == 1:
                    noise = self.model.noise.exp()
                    x = out
                    loss = self.loss_function(x, y, noise)
                else:
                    loss = self.loss_function(out[:, :1], y, out[:, 1:].exp())  # /x.shape[0]
            else:
                loss = self.loss_function(out, y)
                out = torch.softmax(out, -1).mean(0)
                out = out.argmax(dim=-1)

            train_pred.extend(out.tolist())

            tot_loss = mmd + loss

            losses.append(tot_loss.item())
            tot_loss.backward()

            self.optimizer.step()

            progress_bar.set_postfix(ce_loss=loss.item(), mmd_loss=mmd.item())

        return losses, (train_true, train_pred)

    def train_step(self, train_samples=1, test_samples=1, **kwargs):
        losses, train_res = self.train_epoch(samples=train_samples, **kwargs)
        test_res = self.test_evaluation(samples=test_samples, **kwargs)
        return losses, train_res, test_res

    def snr_test(self, percentiles: list):
        return None
