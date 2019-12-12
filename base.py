import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

#
# def pairwise_distances(x, y):
#     x_norm = (x ** 2).sum(1).view(-1, 1)
#     y_norm = (y ** 2).sum(1).view(1, -1)
#
#     dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
#     return torch.clamp(dist, 0.0, np.inf)
# #
# #
# def compute_kernel(x, y):
#     dim = x.size(1)
#     # d = torch.exp(- torch.mul(torch.cdist(x, y).mean(1), 1/float(dim))).mean()
#     d = torch.exp(- torch.mul(pairwise_distances(x, y).mean(1), 1 / float(dim))).mean()
#     return d
# #
# #
# def compute_mmd(x, y):
#     x_kernel = compute_kernel(x, x)
#     y_kernel = compute_kernel(y, y)
#     xy_kernel = compute_kernel(x, y)
#     return x_kernel + y_kernel - 2 * xy_kernel


from bayesian_utils import BayesianCNNLayer, BayesianLinearLayer


def get_bayesian_network(topology, input_image, classes, mu_init, rho_init, prior, divergence, local_trick):
    features = torch.nn.ModuleList()
    # self._prior = prior

    prev = input_image.shape[0]
    input_image = input_image.unsqueeze(0)

    for j, i in enumerate(topology):

        if isinstance(i, (tuple, list)) and i[0] == 'MP':
            l = torch.nn.MaxPool2d(i[1])
            input_image = l(input_image)
            prev = input_image.shape[1]

        elif isinstance(i, float):
            l = torch.nn.Dropout(p=0.5)

        elif isinstance(i, (tuple, list)) and i[0] == 'AP':
            l = torch.nn.AvgPool2d(i[1])
            input_image = l(input_image)
            prev = input_image.shape[1]

        elif isinstance(i, (tuple, list)):
            size, kernel_size = i
            l = BayesianCNNLayer(in_channels=prev, kernels=size, kernel_size=kernel_size,
                                 mu_init=mu_init, divergence=divergence, local_rep_trick=local_trick,
                                 rho_init=rho_init, prior=prior)

            input_image = l(input_image)[0]
            prev = input_image.shape[1]

        elif isinstance(i, int):
            if j > 0 and not isinstance(topology[j - 1], int):
                input_image = torch.flatten(input_image, 1)
                prev = input_image.shape[-1]
                features.append(Flatten())

            size = i
            l = BayesianLinearLayer(in_size=prev, out_size=size, mu_init=mu_init, divergence=divergence,
                                    rho_init=rho_init, prior=prior, local_rep_trick=local_trick)
            prev = size

        else:
            raise ValueError('Topology should be tuple for cnn layers, formatted as (num_kernels, kernel_size), '
                             'pooling layer, formatted as tuple ([\'MP\', \'AP\'], kernel_size, stride) '
                             'or integer, for linear layer. {} was given'.format(i))

        features.append(l)

    if isinstance(topology[-1], (tuple, list)):
        input_image = torch.flatten(input_image, 1)
        prev = input_image.shape[-1]
        features.append(Flatten())

    features.append(BayesianLinearLayer(in_size=prev, out_size=classes, mu_init=mu_init, rho_init=rho_init,
                                        prior=prior, divergence=divergence, local_rep_trick=local_trick))
    return features


def get_network(topology, input_image, classes):
    features = torch.nn.ModuleList()

    prev = input_image.shape[0]
    input_image = input_image.unsqueeze(0)

    for j, i in enumerate(topology):

        if isinstance(i, (tuple, list)) and i[0] == 'MP':
            l = torch.nn.MaxPool2d(i[1])
            input_image = l(input_image)
            prev = input_image.shape[1]

        elif isinstance(i, float):
            l = torch.nn.Dropout(p=0.5)

        elif isinstance(i, (tuple, list)) and i[0] == 'AP':
            l = torch.nn.AvgPool2d(i[1])
            input_image = l(input_image)
            prev = input_image.shape[1]

        elif isinstance(i, (tuple, list)):
            size, kernel_size = i
            l = torch.nn.Conv2d(in_channels=prev, out_channels=size, kernel_size=kernel_size)

            input_image = l(input_image)
            prev = input_image.shape[1]

        elif isinstance(i, int):
            if j > 0 and not isinstance(topology[j - 1], int):
                input_image = torch.flatten(input_image, 1)
                prev = input_image.shape[-1]
                features.append(Flatten())

            size = i
            l = torch.nn.Linear(prev, i)
            prev = size

        else:
            raise ValueError('Topology should be tuple for cnn layers, formatted as (num_kernels, kernel_size), '
                             'pooling layer, formatted as tuple ([\'MP\', \'AP\'], kernel_size, stride) '
                             'or integer, for linear layer. {} was given'.format(i))

        features.append(l)

    if isinstance(topology[-1], (tuple, list)):
        input_image = torch.flatten(input_image, 1)
        prev = input_image.shape[-1]
        features.append(Flatten())

    features.append(torch.nn.Linear(prev, classes))

    return features


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


# PRIORS


class Gaussian(object):
    def __init__(self, mu=0, sigma=5):
        self.mu = mu
        self.sigma = sigma
        self.inner_gaussian = Normal(mu, sigma)
        # self.gaussian = torch.distributions.Normal(mu, sigma)

    def sample(self, size):
        return self.inner_gaussian.rsample(size)
        # return self.mu + self.sigma * Normal(0, 1).sample(size)

    def log_prob(self, x):
        return self.inner_gaussian.log_prob(x)


class ScaledMixtureGaussian(object):
    def __init__(self, pi, s1, s2, mu1=0, mu2=0):
        self.pi = pi
        self.s1 = s1
        self.s2 = s2
        self.mu1 = mu1
        self.mu2 = mu2
        self.gaussian1 = Gaussian(mu1, s1)
        self.gaussian2 = Gaussian(mu2, s2)

    def sample(self, size):
        return self.pi * self.gaussian1.sample(size) + (1 - self.pi) * self.gaussian2.sample(size)

    def log_prob(self, x):
        return self.pi * self.gaussian1.log_prob(x) + (1 - self.pi) * self.gaussian2.log_prob(x)


# Utils

class Network(nn.Module, ABC):

    @abstractmethod
    def layers(self):
        pass

    @abstractmethod
    def eval_forward(self, x, **kwargs):
        pass


class Wrapper(ABC):
    def __init__(self, model: nn.Module, train_data, test_data, optimizer):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.device = next(model.parameters()).device

    def train_step(self, **kwargs):
        losses, train_res = self.train_epoch(**kwargs)
        test_res = self.test_evaluation(**kwargs)
        return losses, train_res, test_res

    @abstractmethod
    def train_epoch(self, **kwargs) -> Tuple[list, Tuple[list, list]]:
        pass

    @abstractmethod
    def test_evaluation(self, **kwargs) -> Tuple[list, list]:
        pass

    @abstractmethod
    def snr_test(self, percentiles: list) -> list:
        pass
