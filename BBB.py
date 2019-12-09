from itertools import chain

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from base import Gaussian, Network, Wrapper, Flatten, get_bayesian_network
from bayesian_utils import BayesianCNNLayer, BayesianLinearLayer
import torch


class BBB(Network):

    def __init__(self, sample, classes, topology=None, prior=None, mu_init=None, rho_init=None,
                 local_trick=False, **kwargs):
        super().__init__()

        if mu_init is None:
            mu_init = (-0.6, 0.6)

        if rho_init is None:
            rho_init = -6

        if topology is None:
            topology = [400, 400]

        if prior is None:
            prior = Gaussian(0, 10)

        self._prior = prior
        self.features = get_bayesian_network(topology, sample, classes,
                                             mu_init, rho_init, prior, 'kl', local_trick)
        # self.features = torch.nn.ModuleList()
        #
        # prev = input_size
        # input_image = input_image.unsqueeze(0)
        #
        # for j, i in enumerate(topology):
        #
        #     if isinstance(i, (tuple, list)) and i[0] == 'MP':
        #         l = torch.nn.MaxPool2d(i[1])
        #         input_image = l(input_image)
        #         prev = input_image.shape[1]
        #
        #     elif isinstance(i, (tuple, list)) and i[0] == 'AP':
        #         l = torch.nn.AvgPool2d(i[1])
        #         input_image = l(input_image)
        #         prev = input_image.shape[1]
        #
        #     elif isinstance(i, (tuple, list)):
        #         size, kernel_size = i
        #         l = BayesianCNNLayer(in_channels=prev, kernels=size, kernel_size=kernel_size,
        #                              mu_init=mu_init, divergence='kl', local_rep_trick=local_trick,
        #                              rho_init=rho_init, prior=self._prior)
        #
        #         input_image = l(input_image)[0]
        #         prev = input_image.shape[1]
        #
        #     elif isinstance(i, int):
        #         if j > 0 and not isinstance(topology[j-1], int):
        #             input_image = torch.flatten(input_image, 1)
        #             prev = input_image.shape[-1]
        #             self.features.append(Flatten())
        #
        #         size = i
        #         l = BayesianLinearLayer(in_size=prev, out_size=size, mu_init=mu_init, divergence='kl',
        #                                 rho_init=rho_init, prior=self._prior, local_rep_trick=local_trick)
        #         prev = size
        #
        #     else:
        #         raise ValueError('Topology should be tuple for cnn layers, formatted as (num_kernels, kernel_size), '
        #                          'pooling layer, formatted as tuple ([\'MP\', \'AP\'], kernel_size) '
        #                          'or integer, for linear layer. {} was given'.format(i))
        #
        #     self.features.append(l)
        #
        # if isinstance(topology[-1], (tuple, list)):
        #     input_image = torch.flatten(input_image, 1)
        #     prev = input_image.shape[-1]
        #     self.features.append(Flatten())
        #
        # self.features.append(BayesianLinearLayer(in_size=prev, out_size=classes, mu_init=mu_init, rho_init=rho_init,
        #                                          prior=self._prior, divergence='kl', local_rep_trick=local_trick))

    def _forward(self, x):

        tot_prior = 0
        tot_post = 0

        for j, i in enumerate(self.features):
            if not isinstance(i, (BayesianLinearLayer, BayesianCNNLayer)):
                x = i(x)
            else:
                x, prior, post = i(x)
                tot_post += post
                tot_prior += prior

            if j < len(self.features)-1:
                x = torch.relu(x)

        #     x = torch.relu(x)
        #
        # if self.ann_type == 'cnn':
        #     x = torch.flatten(x, 1)
        #
        # for i in self.classificator:
        #     x, prior, post = i(x)
        #     tot_post += post
        #     tot_prior += prior

        return x, tot_prior, tot_post

    def forward(self, x, samples=1):
        o = []
        log_priors = []
        log_posts = []

        for i in range(samples):
            op, prior, post = self._forward(x)
            o.append(op)
            log_priors.append(prior)
            log_posts.append(post)

        o = torch.stack(o)
        log_priors = torch.stack(log_priors)
        log_posts = torch.stack(log_posts)

        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        return o, log_prior, log_post

    def layers(self):
        return chain(self.features, self.classificator)

    def eval_forward(self, x, samples=1):
        o, _, _ = self(x, samples=samples)
        return o


class Trainer(Wrapper):
    def __init__(self, model: nn.Module, train_data, test_data, optimizer):
        super().__init__(model, train_data, test_data, optimizer)

    def train_epoch(self, samples=1, **kwargs):
        losses = []

        M = len(self.train_data)
        a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
        b = 2 ** M - 1

        pi = a / b

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=True)
        # progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

        train_true = []
        train_pred = []

        for batch, (x_train, y_train) in progress_bar:
            train_true.extend(y_train.tolist())

            self.optimizer.zero_grad()

            out, prior, post = self.model(x_train.to(self.device), samples=samples)
            out = out.mean(0)
            logloss = (- post - prior) * pi[batch]  # /x_train.shape[0]

            ce = F.cross_entropy(out, y_train.to(self.device), reduction='sum')
            loss = ce + logloss
            losses.append(loss.item())
            loss.backward()

            self.optimizer.step()

            max_class = F.log_softmax(out, -1).argmax(dim=-1)
            train_pred.extend(max_class.tolist())
            # progress_bar.set_postfix(ce_loss=loss.item())

        return losses, (train_true, train_pred)

    def test_evaluation(self, samples, **kwargs):

        test_pred = []
        test_true = []

        self.model.eval()
        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(self.test_data):
                test_true.extend(y_test.tolist())

                out = self.model.eval_forward(x_test.to(self.device), samples=samples)
                out = out.mean(0)
                out = out.argmax(dim=-1)
                test_pred.extend(out.tolist())

        return test_true, test_pred

    def train_step(self, train_samples=1, test_samples=1, **kwargs):
        losses, train_res = self.train_epoch(samples=train_samples)
        test_res = self.test_evaluation(samples=test_samples)
        return losses, train_res, test_res

    def snr_test(self, percentiles: list):
        return None
