from torch import nn as nn
from torch.distributions import Normal
from torch.nn import Parameter, functional as F
import torch


class BayesianCNNLayer(nn.Module):
    def __init__(self, in_channels, kernels, divergence, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 mu_init=None, rho_init=None, local_rep_trick=False, prior=None, **kwargs):

        super().__init__()

        divergence = divergence.lower()
        if divergence not in ['mmd', 'kl']:
            raise ValueError('type parameter should be mmd or kl.')

        self.divergence = divergence

        self.in_channels = in_channels
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.local_trick = local_rep_trick
        self.w = BayesianParameters(size=(kernels, in_channels, kernel_size, kernel_size),
                                    mu_initialization=mu_init, rho_initialization=rho_init)

        if local_rep_trick:
            self.log_alpha = Parameter(torch.Tensor(1, 1))

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
        if not self.local_trick:
            w = self.w.weights
            o = F.conv2d(x, weight=w)
            return o, w, b
        else:
            w_mu = F.conv2d(x, weight=self.w.mu)
            w_std = torch.sqrt(1e-12 + F.conv2d(x.pow(2), weight=torch.exp(self.log_alpha) * self.w.mu.pow(2)))

            output = w_mu + w_std * torch.randn(w_std.size()).to(w_std.device)

            return output, self.w.weights, b

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

        if local_rep_trick:
            self.log_alpha = Parameter(torch.Tensor(1, 1))

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

            w_mu = F.linear(input=x, weight=self.w.mu)
            w_std = torch.sqrt(1e-12 + F.linear(input=x.pow(2), weight=torch.exp(self.log_alpha) * self.w.mu.pow(2)))

            w_out = w_mu + w_std * torch.randn(w_mu.shape, requires_grad=True).to(x.device)

            if self.b is not None:
                b = self.b.weights
                w_out += b.unsqueeze(0).expand(x.shape[0], -1)

            return w_out, self.w.weights, b

    def _mmd_forward(self, x):
        o, w, b = self._forward(x)

        mmd_w = torch.tensor(0.0)  # .float()
        mmd_b = torch.tensor(0.0)  # .float()

        if self.training:
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device))

            if b is not None:
                b = b.unsqueeze(0)
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


def b_drop(x, p=0.5):
    return F.dropout(x, p=p, training=True, inplace=False)


class BayesDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True, inplace=False)


def compute_mmd(x, y):
    # dim = x.size(1)
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * xy

    # print(xx.device)
    XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device))

    for a in [0.05, 0.1, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    # # for a in [dim]:
    # XX = (torch.exp(-xx) * dim ** -1).mean()
    # YY = (torch.exp(-yy) * dim ** -1).mean()
    # XY = (torch.exp(-xy) * dim ** -1).mean()

    return torch.mean(XX + YY - 2. * XY)
