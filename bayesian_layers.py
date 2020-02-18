from abc import ABC, abstractmethod

import torch
from torch import nn as nn
from torch.nn import functional as F

from bayesian_utils import compute_mmd, BayesianParameters


class BayesianLayer(ABC, nn.Module):
    def __init__(self, divergence, prior, local_trick=False, **kwargs):
        super().__init__()

        divergence = divergence.lower()
        if divergence not in ['mmd', 'kl']:
            raise ValueError('type parameter should be mmd or kl.')

        if divergence == 'mmd':
            self.biased = kwargs.get('biased', False)
            self.kernel = kwargs.get('kernel', 'inverse')
            if self.kernel not in ['rbf', 'inverse']:
                raise ValueError('Available kernels: rbf, inverse. {} given.'.format(self.kernel))

            self.alpha = kwargs.get('alpha', None)
            if self.alpha is not None and (self.alpha > 1 or self.alpha < 0):
                raise ValueError('Alpha should be between 0 and 1 (or None), {} given.'.format(self.alpha))

        self.divergence = divergence
        self.local_trick = local_trick

        self.w = None
        self.b = None
        self.w_w = None
        self.b_w = None

        self.prior_w = prior
        self.prior_b = prior
        self.log_prior = None
        self.log_posterior = None

    def _mmd_forward(self, x, calculate_divergence):
        o, w, b = self._forward(x)

        mmd_w = torch.tensor(0.0).to(x.device)  # .float()
        mmd_b = torch.tensor(0.0).to(x.device)  # .float()

        if self.training and calculate_divergence:
            w = torch.flatten(w, 1)
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device), type=self.kernel, biased=self.biased)

            if b is not None:
                b = b.unsqueeze(0)
                mmd_b = compute_mmd(b, self.prior_b.sample(b.size()).to(w.device), type=self.kernel,
                                    biased=self.biased)

            if self.alpha is not None:
                if torch.abs(mmd_b) < self.alpha:
                    mmd_b = torch.tensor(0.0).to(x.device)
                if torch.abs(mmd_w) < self.alpha:
                    mmd_w = torch.tensor(0.0).to(x.device)

        return o, mmd_w + mmd_b

    def _kl_forward(self, x, calculate_divergence):
        o, w, b = self._forward(x)
        log_post = torch.tensor(0.0)
        log_prior = torch.tensor(0.0)

        if self.training and calculate_divergence:
            log_post = self.w.posterior_log_prob(w).sum()
            log_prior = self.prior_w.log_prob(w).sum()

            if b is not None:
                log_post += self.b.posterior_log_prob(b).sum()
                log_prior += self.prior_b.log_prob(b).sum()

        return o, log_prior, log_post

    def forward(self, x, calculate_divergence=True):
        if self.divergence == 'kl':
            return self._kl_forward(x, calculate_divergence)
        if self.divergence == 'mmd':
            return self._mmd_forward(x, calculate_divergence)

    def set_mask(self, p):
        self.w.set_mask(p)
        if self.b is not None:
            self.b.set_mask(p)

    def prior_prob(self, prior: torch.distributions = None, log=True):

        if prior is None:
            prior = self.prior_w

        w_log = prior.log_prob(self.w.weights).sum()
        b_log = prior.log_prob(self.b.weights).sum() if self.b is not None else 1

        if not log:
            w_log = w_log.exp()
            b_log = b_log.exp() if self.b is not None else 1

        return w_log + b_log

    @abstractmethod
    def _forward(self, x):
        pass


class BayesianCNNLayer(BayesianLayer):
    def __init__(self, in_channels, kernels, divergence, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 mu_init=None, rho_init=None, local_rep_trick=False, prior=None, posterior_type='weights', **kwargs):
        super().__init__(divergence, prior, local_rep_trick, **kwargs)

        self.in_channels = in_channels
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.w = BayesianParameters(size=(kernels, in_channels, kernel_size, kernel_size),
                                    posterior_type=posterior_type,
                                    mu_initialization=mu_init, rho_initialization=rho_init)

    def _forward(self, x):

        b = None
        if not self.local_trick:
            w = self.w.weights
            o = F.conv2d(x, weight=w, stride=self.stride, padding=self.padding)
            return o, w, b
        else:
            w_mu = F.conv2d(x, weight=self.w.mu)
            w_std = torch.sqrt(1e-12 + F.conv2d(x.pow(2), weight=self.w.sigma))

            output = w_mu + w_std * torch.randn(w_std.size(), requires_grad=True).to(w_std.device)

            return output, self.w.weights, b

    def extra_repr(self):
        return 'input: {}, output: {}, kernel_size: {}, bias: {}'.format(self.in_channels, self.kernels,
                                                                         self.kernel_size,
                                                                         True if self.b is not None else False)


class BayesianLinearLayer(BayesianLayer):
    def __init__(self, in_size, out_size, divergence, mu_init=None, rho_init=None, use_bias=True, prior=None,
                 local_rep_trick=False, posterior_type='weights', **kwargs):

        super().__init__(divergence, prior, local_rep_trick, **kwargs)

        self.in_size = in_size
        self.out_size = out_size

        self.w = BayesianParameters(size=(out_size, in_size), posterior_type=posterior_type,
                                    mu_initialization=mu_init, rho_initialization=rho_init)

        self.b = None
        if use_bias:
            self.b = BayesianParameters(size=out_size, mu_initialization=mu_init, is_bias=True,
                                        rho_initialization=rho_init, posterior_type=posterior_type)

    def _forward(self, x):
        b = None
        if not self.local_trick:
            w = self.w.weights
            if self.b is not None:
                b = self.b.weights
            o = F.linear(x, w, b)
            return o, w, b
        else:
            w_mu = F.linear(input=x, weight=self.w.mu)

            w_std = torch.sqrt(1e-12 + F.linear(input=x.pow(2), weight=self.w.sigma))

            w_out = w_mu + w_std * torch.randn(w_mu.shape, requires_grad=True).to(x.device)

            if self.b is not None:
                b = self.b.weights
                w_out += b.unsqueeze(0).expand(x.shape[0], -1)

            return w_out, self.w.weights, b

    def extra_repr(self):
        return 'input: {}, output: {}, bias: {}'.format(self.in_size, self.out_size,
                                                        True if self.b is not None else False)


class BayesDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True, inplace=False)