from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from base import Wrapper, get_bayesian_network, Network
from priors import Gaussian
from bayesian_layers import BayesianCNNLayer, BayesianLinearLayer


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
    def __init__(self, model: nn.Module, train_data, test_data, optimizer, wd=None):
        super().__init__(model, train_data, test_data, optimizer)
        self.wd = wd

    def weight_normalization_loss(self):
        l2_reg = torch.tensor(0.).to(self.device)
        for j, i in enumerate(self.model.features):
            if isinstance(i, (BayesianLinearLayer, BayesianCNNLayer)):
                l2_reg += torch.norm(i.w.weights)
                if i.b is not None:
                    l2_reg += torch.norm(i.b.weights)
        return l2_reg

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

            mmd *= mmd_w
            mmd *= pi[batch]

            if pi[batch] == 0:
                self.model.calculate_mmd = False

            if self.regression:
                out = out.mean(0)
                if self.model.classes == 1:
                    noise = self.model.noise.exp()
                    x = out
                    loss = self.loss_function(x, y, noise)
                else:
                    loss = self.loss_function(out[:, :1], y, out[:, 1:].exp())
                loss = loss/x.shape[0]
            else:
                loss = self.loss_function(out, y)
                out = torch.softmax(out, -1).mean(0)
                out = out.argmax(dim=-1)

            train_pred.extend(out.tolist())

            tot_loss = mmd + loss

            if self.wd is not None:
                reg = self.weight_normalization_loss() * self.wd
                tot_loss += reg

            losses.append(tot_loss.item())
            tot_loss.backward()

            self.optimizer.step()

            progress_bar.set_postfix(ce_loss=loss.item(), mmd_loss=mmd.item())

        return losses, (train_true, train_pred)

    def train_step(self, train_samples=1, test_samples=1, **kwargs):
        losses, train_res = self.train_epoch(samples=train_samples, **kwargs)
        test_res = self.test_evaluation(samples=test_samples, **kwargs)
        return losses, train_res, test_res
