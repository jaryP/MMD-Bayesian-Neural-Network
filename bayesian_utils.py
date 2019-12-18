import numpy as np

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

        if mu_init is None:
            input_size = kernels * np.power(in_channels, 2)
            # std = 1/np.sqrt(std)
            std = np.sqrt(2/input_size)

            # std *= 3
            mu_init = (-std, std)
            # rho_init = np.log(np.exp(2/(3*input_size)) - 1)
            # print(mu_init, rho_init)

        if rho_init is None:
            rho_init = -3
        # input_size = kernels * np.power(in_channels, 2)
        # std = np.sqrt(2/input_size)
        # # mu_init = [-std, std]
        # # rho_init = -3
        #
        # mu_init = [-std, std]
        # rho_init = -3

        self.divergence = divergence

        self.in_channels = in_channels
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.local_trick = local_rep_trick
        self.w = BayesianParameters(size=(kernels, in_channels, kernel_size, kernel_size),
                                    mu_initialization=mu_init, rho_initialization=rho_init)

        if local_rep_trick:
            self.log_alpha = nn.Parameter(torch.zeros((1, 1)).uniform_(*mu_init))

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

            output = w_mu + w_std * torch.randn(w_std.size(), requires_grad=True).to(w_std.device)

            return output, self.w.weights, b

    def _mmd_forward(self, x, calculate_divergence):
        o, w, b = self._forward(x)

        mmd_w = torch.tensor(0.0).to(x.device)  # .float()
        mmd_b = torch.tensor(0.0).to(x.device)  # .float()

        if self.training and calculate_divergence:
            w = torch.flatten(w, 1)
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device))

            if b is not None:
                b = b.unsqueeze(0)
                mmd_b = compute_mmd(b, self.prior_b.sample(b.size()).to(b.device))

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

    def extra_repr(self):
        return 'input: {}, output: {}, kernel_size: {}, bias: {}'.format(self.in_channels, self.kernels,
                                                                         self.kernel_size,
                                                                         True if self.b is not None else False)


class BayesianLinearLayer(nn.Module):
    def __init__(self, in_size, out_size, divergence, mu_init=None, rho_init=None, use_bias=True, prior=None,
                 local_rep_trick=False):

        super().__init__()

        if mu_init is None:
            #     # std = 1/np.sqrt(in_size)
            #     # std *= 3
            #     std = np.sqrt(2/in_size)*3
            #     mu_init = (-std, std)
            #     print(mu_init)

            # std = 1/np.sqrt(std)
            std = np.sqrt(2/in_size)

            # std *= 3
            mu_init = (-std, std)
            # rho_init = np.log(np.exp(2/(3*in_size)) - 1)

        if rho_init is None:
            rho_init = -3

        # std = np.sqrt(2/in_size)
        # mu_init = [-std, std]
        # rho_init = -3

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
            self.log_alpha = nn.Parameter(torch.zeros((1, 1)).uniform_(*mu_init))

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

    def _mmd_forward(self, x, calculate_divergence):
        o, w, b = self._forward(x)

        mmd_w = torch.tensor(0.0)  # .float()
        mmd_b = torch.tensor(0.0)  # .float()

        if self.training and calculate_divergence:
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device))

            if b is not None:
                b = b.unsqueeze(0)
                mmd_b = compute_mmd(b, self.prior_w.sample(b.size()).to(w.device))

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

    def forward(self, x, calculate_divergence=True):
        if self.divergence == 'kl':
            return self._kl_forward(x, calculate_divergence)
        if self.divergence == 'mmd':
            return self._mmd_forward(x, calculate_divergence)

    def extra_repr(self):
        return 'input: {}, output: {}, bias: {}'.format(self.in_size, self.out_size,
                                                        True if self.b is not None else False)


class BayesianParameters(nn.Module):
    def __init__(self, size, mu_initialization=None, rho_initialization=None):
        super().__init__()

        if mu_initialization is None:
            self.mu = nn.Parameter(torch.randn(size))
        elif isinstance(mu_initialization, (list, tuple)):
            if len(mu_initialization) != 2:
                raise ValueError("If you want to initialize mu uniformly,"
                                 " mu_init should have len=2, {} given".format(mu_initialization))
            self.mu = nn.Parameter(torch.zeros(size).uniform_(*mu_initialization))
        elif isinstance(mu_initialization, (float, int)):
            self.mu = nn.Parameter(torch.ones(size) * mu_initialization)
        else:
            raise ValueError("Error mu")

        # self.mu = self.mu.div(self.mu.norm(p=2, dim=0, keepdim=True))

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
        sigma = self.sigma
        r = self.mu + sigma * torch.randn(self.mu.shape, requires_grad=True).to(self.mu.device)
        return r

    @property
    def sigma(self):
        return F.softplus(self.rho)
        # return torch.log(1 + torch.exp(self.rho))

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


def pdist(p, q):
    pdim, qdim = p.size(0), q.size(0)
    pnorm = torch.sum(p**2, dim=1, keepdim=True)
    qnorm = torch.sum(q**2, dim=1, keepdim=True)
    norms = (pnorm.expand(pdim, qdim) +
             qnorm.transpose(0, 1).expand(pdim, qdim))
    distances_squared = norms - 2 * p.mm(q.t())
    return torch.sqrt(1e-5 + torch.abs(distances_squared))


def pairwise_distances(x, y):

    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def compute_mmd(x, y, type='inverse'):
    if type == 'rbf':
        xxd = pdist(x, x)**2
        yyd = pdist(y, y)**2
        xyd = pdist(x, y)**2

        xs, ys = xxd.shape[0], yyd.shape[0]

        gamma = 1/xs
        xx = torch.exp(-xxd/gamma)
        xx = xx.sum() - xx.trace()

        yy = torch.exp(-yyd/gamma)
        yy = yy.sum() - yy.trace()

        xy = torch.exp(-xyd/gamma).sum()

        mmd = (2 / (ys * xs)) * xy + (1 / (xs ** 2)) * xx + (1 / (ys ** 2)) * yy
        return mmd

    if type == 'inverse':
        xxd = pdist(x, x)**2
        yyd = pdist(y, y)**2
        xyd = pdist(x, y)**2

        xs, ys = xxd.shape[0], yyd.shape[0]

        XX, YY, XY = 0, 0, 0

        for a in [0.05, 0.2, 0.9]:
            a = a ** 2
            xxk = a * ((a + xxd) ** -1)
            yyk = a * ((a + yyd) ** -1)
            xyk = a * ((a + xyd) ** -1)

            xxk = xxk.sum() - xxk.trace()
            yyk = yyk.sum() - yyk.trace()
            xyk = xyk.sum()

            XX += xxk
            YY += yyk
            XY += xyk

        mmd = (2 / (ys * xs)) * XY + (1 / (xs ** 2)) * XX + (1 / (ys ** 2)) * YY
        return mmd

    return None
