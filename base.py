from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm
import torch.nn.functional as F


# def pairwise_distances(x, y):
#     x_norm = (x ** 2).sum(1).view(-1, 1)
#     y_norm = (y ** 2).sum(1).view(1, -1)
#
#     dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
#     return torch.clamp(dist, 0.0, np.inf)
#
#
# def compute_kernel(x, y):
#     dim = x.size(1)
#     # d = torch.exp(- torch.mul(torch.cdist(x, y).mean(1), 1/float(dim))).mean()
#     d = torch.exp(- torch.mul(pairwise_distances(x, y).mean(1), 1 / float(dim))).mean()
#     return d
#
#
# def compute_mmd(x, y):
#     x_kernel = compute_kernel(x, x)
#     y_kernel = compute_kernel(y, y)
#     xy_kernel = compute_kernel(x, y)
#     return x_kernel + y_kernel - 2 * xy_kernel


def compute_mmd(x, y):
    dim = x.size(1)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    # XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
    #               torch.zeros(xx.shape).to(x.device),
    #               torch.zeros(xx.shape).to(x.device))
    # for a in [0.1]:
    #     XX += a**2 * (a**2 + dxx)**-1
    #     YY += a**2 * (a**2 + dyy)**-1
    #     XY += a**2 * (a**2 + dxy)**-1

    # for a in [dim]:
    XX = torch.exp(-dxx.to(x.device)) * dim ** -1
    YY = torch.exp(-dyy.to(x.device)) * dim ** -1
    XY = torch.exp(-dxy.to(x.device)) * dim ** -1

    return torch.mean(XX + YY - 2. * XY)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class BayesianParameters(nn.Module):
    def __init__(self, size, mu_initialization=None, rho_initialization=None):
        super().__init__()

        if mu_initialization is None:
            self.mu = nn.Parameter(torch.randn(size))
        elif isinstance(mu_initialization, (list, tuple)):
            self.mu = nn.Parameter(torch.zeros(size).uniform_(*mu_initialization))
        elif isinstance(mu_initialization, (float, int)):
            self.mu = nn.Parameter(torch.ones(size) * mu_initialization)
        else:
            raise ValueError("Error mu")

        if rho_initialization is None:
            self.rho = nn.Parameter(torch.randn(size))
        elif isinstance(rho_initialization, (list, tuple)):
            self.rho = nn.Parameter(torch.zeros(size).uniform_(*rho_initialization))
        elif isinstance(rho_initialization, (float, int)):
            self.rho = nn.Parameter(torch.ones(size) * rho_initialization)
        else:
            raise ValueError("Error rho")

    @property
    def weights(self):
        return self.mu + self.sigma * Normal(0, 1).sample(self.mu.shape).to(self.mu.device)

    @property
    def sigma(self):
        return torch.log(1 + torch.exp(self.rho))

    def prior(self, prior: torch.distributions, w, log=True):
        if log:
            return prior.log_prob(w)
        else:
            return prior.prob(w)

    def posterior_distribution(self):
        return Normal(self.mu.data.clone(), torch.log(1 + torch.exp(self.rho)).clone())

    def posterior_log_prob(self, w):
        return self.posterior_distribution().log_prob(w)

    def forward(self, input, sample=1):
        pass

    # def extra_repr(self):
    #     return 'BayesianParameters(mu: {}, rho: {})'.format(tuple(self.mu.shape), tuple(self.rho.shape))


class BayesianLinearLayer(nn.Module):
    def __init__(self, in_size, out_size, divergence, mu_init=None, rho_init=None, use_bias=True, prior=None,
                 local_rep_trick=False):

        super().__init__()

        divergence = divergence.lower()
        if divergence not in ['mmd', 'kl']:
            raise ValueError('type parameter should be mmd or kl.')

        self.local_trick = local_rep_trick

        self.divergence = divergence

        self.in_size = in_size
        self.out_size = out_size

        self.w = BayesianParameters(size=(out_size, in_size),
                                    mu_initialization=mu_init, rho_initialization=rho_init)

        self.b = None
        if use_bias:
            self.b = BayesianParameters(size=out_size,
                                        mu_initialization=mu_init, rho_initialization=rho_init)

        self.w_w = None
        self.b_w = None

        self.prior_w = prior
        self.prior_b = prior
        self.log_prior = None
        self.log_posterior = None

    def _forward(self, x):
        b = None
        if not self.local_trick:
            w = self.w.weights
            if self.b is not None:
                b = self.b.weights
            o = F.linear(x, w, b)
            return o, w, b
        else:
            sigma_w = self.w.sigma
            w_mu = torch.mm(x, self.w.mu.t())
            w_std = torch.sqrt(torch.mm(x.pow(2), sigma_w.pow(2).t()))

            w_out = w_mu + w_std * torch.randn(w_mu.shape, requires_grad=True).to(x.device)

            if self.b is not None:
                b = self.b.weights
                w_out += b.unsqueeze(0).expand(x.shape[0], -1)
                # torch.randn(w_mu.shape, requires_grad=True).to(self.mu.device)

            return w_out, self.w.weights, b

    def _mmd_forward(self, x):
        o, w, b = self._forward(x)

        mmd_w = torch.tensor(0.0)  # .float()
        mmd_b = torch.tensor(0.0)  # .float()

        if self.training:
            # mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device))
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device))

            if b is not None:
                b = b.unsqueeze(0)
                # mmd_b = compute_mmd(b, self.prior_b.sample(b.size()).to(b.device))
                mmd_b = compute_mmd(b, self.prior_w.sample(b.size()).to(w.device))

        return o, mmd_w + mmd_b

    def _kl_forward(self, x):
        o, w, b = self._forward(x)
        log_post = torch.tensor(0.0)
        log_prior = torch.tensor(0.0)

        if self.training:
            log_post = self.w.posterior_log_prob(w).sum()
            log_prior = self.prior_w.log_prob(w).sum()

            if b is not None:
                log_post += self.b.posterior_log_prob(b).sum()
                log_prior += self.prior_b.log_prob(b).sum()

        return o, log_prior, log_post

    # def _mmd_forward(self, x):
    #     w = self.w.weights
    #
    #     b = None
    #     mmd_w = torch.tensor(0).float()
    #     mmd_b = torch.tensor(0).float()
    #
    #     if self.training:
    #         mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device))
    #
    #         if self.b is not None:
    #             b = self.b.weights.unsqueeze(0)
    #             mmd_b = compute_mmd(b, self.prior_b.sample(b.size()).to(b.device))
    #
    #     o = F.linear(x, w, b)
    #
    #     self.w_w = w
    #     self.b_w = b
    #
    #     return o, mmd_w + mmd_b

    # def _kl_forward(self, x):
    #     w = self.w.weights
    #     log_post = self.w.posterior_log_prob(w).sum()
    #     log_prior = self.prior_w.log_prob(w).sum()
    #     b = None
    #
    #     if self.b is not None:
    #         b = self.b.weights
    #
    #         log_post += self.b.posterior_log_prob(b).sum()
    #         log_prior += self.prior_b.log_prob(b).sum()
    #
    #     o = F.linear(x, w, b)
    #
    #     self.log_prior = log_prior
    #     self.log_posterior = log_post
    #     self.w_w = w
    #     self.b_w = b
    #
    #     return o, log_prior, log_post

    @property
    def weights(self):
        return self.w_w, self.b_w

    def set_prior(self, w=None, b=None):
        if w is not None:
            self.prior_w = w
        if b is not None:
            self.prior_b = b

    def posterior(self):
        return self.w.posterior_distribution(), self.b.posterior_distribution()

    def posterior_distribution(self):
        return self.w.posterior_distribution(), self.b.posterior_distribution()

    def forward(self, x):
        if self.divergence == 'kl':
            return self._kl_forward(x)
        if self.divergence == 'mmd':
            return self._mmd_forward(x)

    def extra_repr(self):
        return 'input: {}, output: {}, bias: {}'.format(self.in_size, self.out_size,
                                                        True if self.b is not None else False)


class BayesianCNNLayer(nn.Module):
    def __init__(self, in_channels, kernels, divergence, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 mu_init=None, rho_init=None, prior=None, **kwargs):

        super().__init__()

        divergence = divergence.lower()
        if divergence not in ['mmd', 'kl']:
            raise ValueError('type parameter should be mmd or kl.')

        self.divergence = divergence

        self.in_channels = in_channels
        self.kernels = kernels
        self.kernel_size = kernel_size

        self.w = BayesianParameters(size=(kernels, in_channels, kernel_size, kernel_size),
                                    mu_initialization=mu_init, rho_initialization=rho_init)

        self.b = None
        # if use_bias:
        #     self.b = BayesianParameters(size=out_size,
        #                                 mu_initialization=mu_init, rho_initialization=rho_init)

        self.w_w = None
        self.b_w = None

        self.prior_w = prior
        self.prior_b = prior
        self.log_prior = None
        self.log_posterior = None

    def _forward(self, x):
        b = None
        # if not self.local_trick:
        w = self.w.weights
        if self.b is not None:
            b = self.b.weights
        o = F.conv2d(x, weight=w)
        return o, w, b
        # else:
        #     assert False
        # sigma_w = self.w.sigma
        # w_mu = torch.mm(x, self.w.mu.t())
        # w_std = torch.sqrt(torch.mm(x.pow(2), sigma_w.pow(2).t()))
        #
        # w_out = w_mu + w_std * torch.randn(w_mu.shape, requires_grad=True).to(x.device)
        #
        # if self.b is not None:
        #     b = self.b.weights
        #     w_out += b.unsqueeze(0).expand(x.shape[0], -1)
        #     # torch.randn(w_mu.shape, requires_grad=True).to(self.mu.device)
        #
        # return w_out, self.w.weights, b

    def _mmd_forward(self, x):
        o, w, b = self._forward(x)

        mmd_w = torch.tensor(0.0).to(x.device)  # .float()
        mmd_b = torch.tensor(0.0).to(x.device)  # .float()

        if self.training:
            w = torch.flatten(w, 1)
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device))

            if b is not None:
                b = b.unsqueeze(0)
                mmd_b = compute_mmd(b, self.prior_b.sample(b.size()).to(b.device))

        return o, mmd_w + mmd_b

    def _kl_forward(self, x):
        o, w, b = self._forward(x)
        log_post = torch.tensor(0.0)
        log_prior = torch.tensor(0.0)

        if self.training:
            log_post = self.w.posterior_log_prob(w).sum()
            log_prior = self.prior_w.log_prob(w).sum()

            if b is not None:
                log_post += self.b.posterior_log_prob(b).sum()
                log_prior += self.prior_b.log_prob(b).sum()

        return o, log_prior, log_post

    def forward(self, x):
        if self.divergence == 'kl':
            return self._kl_forward(x)
        if self.divergence == 'mmd':
            return self._mmd_forward(x)

    def extra_repr(self):
        return 'input: {}, output: {}, kernel_size: {}, bias: {}'.format(self.in_channels, self.kernels,
                                                                        self.kernel_size,
                                                                        True if self.b is not None else False)

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
