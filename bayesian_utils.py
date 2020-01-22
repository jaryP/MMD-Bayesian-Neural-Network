import numpy as np

from torch import nn as nn
from torch.distributions import Normal
from torch.nn import Parameter, functional as F, init
import torch


class BayesianCNNLayer(nn.Module):
    def __init__(self, in_channels, kernels, divergence, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 mu_init=None, rho_init=None, local_rep_trick=False, prior=None, posterior_type='weights', **kwargs):

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

        self.in_channels = in_channels
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.local_trick = local_rep_trick
        self.stride = stride
        self.padding = padding

        self.w = BayesianParameters(size=(kernels, in_channels, kernel_size, kernel_size), posterior_type=posterior_type,
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
        if not self.local_trick:
            w = self.w.weights
            o = F.conv2d(x, weight=w, stride=self.stride, padding=self.padding)
            return o, w, b
        else:
            w_mu = F.conv2d(x, weight=self.w.mu)
            w_std = torch.sqrt(1e-12 + F.conv2d(x.pow(2), weight=self.w.sigma))

            output = w_mu + w_std * torch.randn(w_std.size(), requires_grad=True).to(w_std.device)

            return output, self.w.weights, b

    def _mmd_forward(self, x, calculate_divergence):
        o, w, b = self._forward(x)

        mmd_w = torch.tensor(0.0).to(x.device)  # .float()
        mmd_b = torch.tensor(0.0).to(x.device)  # .float()

        if self.training and calculate_divergence:
            w = torch.flatten(w, 1)
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device), type=self.kernel, biased=self.biased)

            if b is not None:
                b = b.unsqueeze(0)
                mmd_b = compute_mmd(b, self.prior_b.sample(b.size()).to(w.device), type=self.kernel, biased=self.biased)

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

    def extra_repr(self):
        return 'input: {}, output: {}, kernel_size: {}, bias: {}'.format(self.in_channels, self.kernels,
                                                                         self.kernel_size,
                                                                         True if self.b is not None else False)


class BayesianLinearLayer(nn.Module):
    def __init__(self, in_size, out_size, divergence, mu_init=None, rho_init=None, use_bias=True, prior=None,
                 local_rep_trick=False, posterior_type='weights', **kwargs):

        super().__init__()

        if divergence == 'mmd':
            self.biased = kwargs.get('biased', False)
            self.kernel = kwargs.get('kernel', 'inverse')
            if self.kernel not in ['rbf', 'inverse']:
                raise ValueError('Available kernels: rbf, inverse. {} given.'.format(self.kernel))
            self.alpha = kwargs.get('alpha', None)
            if self.alpha is not None and (self.alpha > 1 or self.alpha < 0):
                raise ValueError('Alpha should be between 0 and 1 (or None), {} given.'.format(self.alpha))

        divergence = divergence.lower()
        if divergence not in ['mmd', 'kl']:
            raise ValueError('type parameter should be mmd or kl.')

        self.local_trick = local_rep_trick

        self.divergence = divergence

        self.in_size = in_size
        self.out_size = out_size

        self.w = BayesianParameters(size=(out_size, in_size), posterior_type=posterior_type,
                                    mu_initialization=mu_init, rho_initialization=rho_init)

        self.b = None
        if use_bias:
            self.b = BayesianParameters(size=out_size, mu_initialization=mu_init, is_bias=True,
                                        rho_initialization=rho_init, posterior_type=posterior_type)

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

            w_std = torch.sqrt(1e-12 + F.linear(input=x.pow(2), weight=self.w.sigma))

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
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device), type=self.kernel, biased=self.biased)

            if b is not None:
                b = b.unsqueeze(0)
                mmd_b = compute_mmd(b, self.prior_w.sample(b.size()).to(w.device), type=self.kernel, biased=self.biased)

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

    @property
    def weights(self):
        return self.w_w, self.b_w

    def set_prior(self, w=None, b=None):
        if w is not None:
            self.prior_w = w
        if b is not None:
            self.prior_b = b

    def set_mask(self, p):
        self.w.set_mask(p)
        if self.b is not None:
            self.b.set_mask(p)

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
    def __init__(self, size, mu_initialization=None, rho_initialization=None, posterior_type='weights', is_bias=False):
        super().__init__()
        self.posterior_type = posterior_type
        self.mask = None

        if mu_initialization is None:
            t = torch.empty(size)
            # print(mu_initialization, is_bias, size)
            if is_bias:
                bound = 1 / np.sqrt(size)
                init.uniform_(t, -bound, bound)
            else:
                # nn.init.kaiming_uniform_(t, mode='fan_in', nonlinearity='relu')
                init.kaiming_uniform_(t, a=np.sqrt(5))
            self.mu = nn.Parameter(t, requires_grad=True)

        else:
            t = mu_initialization['type']

            if t == 'uniform':
                a, b = mu_initialization['a'], mu_initialization['b']
                self.mu = nn.Parameter(torch.zeros(size).uniform_(a, b), requires_grad=True)
            elif t == 'gaussian':
                mu, sigma = mu_initialization['mu'], mu_initialization['sigma']
                self.mu = nn.Parameter(torch.zeros(size).normal_(mu, sigma), requires_grad=True)
            elif t == 'constant':
                self.mu = nn.Parameter(torch.ones(size) * mu_initialization['c'], requires_grad=True)
            else:
                raise ValueError("Pissible initialization for mu parameter: \n"
                                 "-gaussian {{mu, sigma}}\n"
                                 "-uniform {{a, b}}\n"
                                 "-constant {{c}}. \n {} was given".format(t))

        rho_size = size

        if posterior_type == 'layers':
            rho_size = (1,)
        elif posterior_type == 'neurons':
            if is_bias:
                rho_size = size
            else:
                rho_size = list(size)
                for i in range(1, len(rho_size)):
                    rho_size[i] = 1
                    #   rho_size = (size[0], 1)

        if rho_initialization is None:
            rho = torch.randn(rho_size)
            # self.rho = nn.Parameter(torch.randn(rho_size), requires_grad=True)
        else:
            t = rho_initialization['type']

            if t == 'uniform':
                a, b = rho_initialization['a'], rho_initialization['b']
                rho = torch.zeros(rho_size).uniform_(a, b)
                # self.rho = nn.Parameter(torch.zeros(rho_size).uniform_(a, b), requires_grad=True)
            elif t == 'gaussian':
                mu, sigma = rho_initialization['mu'], rho_initialization['sigma']
                rho = torch.zeros(rho_size).normal_(mu, sigma)
                # self.rho = nn.Parameter(torch.zeros(rho_size).normal_(mu, sigma), requires_grad=True)
            elif t == 'constant':
                rho = torch.ones(rho_size) * rho_initialization['c']
                # self.rho = nn.Parameter(torch.ones(rho_size) * rho_initialization['c'], requires_grad=True)
            else:
                raise ValueError("Pissible initialization for rho parameter: \n"
                                 "-gaussian {{mu, sigma}}\n"
                                 "-uniform {{a, b}}\n"
                                 "-constant {{c}}. \n {} was given".format(t))

        # if posterior_type != 'weights':
        #     rho = rho/self.mu.pow(2).data

        self.rho = nn.Parameter(rho, requires_grad=True)

    def set_mask(self, p):
        if p is not None:
            if p < 0 or p > 1:
                raise ValueError('Mask percentile should be between 0 and 1, {} was given'.format(p))
        self.mask = p

    @property
    def weights(self):
        sigma = self.sigma
        r = self.mu + sigma * torch.randn(self.mu.shape, requires_grad=True).to(self.mu.device)

        if self.mask is not None:
            mean = torch.abs(self.mu)
            std = self.sigma
            snr = mean / std
            percentile = np.percentile(snr.cpu(), self.mask*100)
            mask = torch.ones_like(snr)
            mask[snr < torch.tensor(percentile)] = 0

            r = r * mask

        return r

    @property
    def sigma(self):
        if self.posterior_type == 'weights':
            return F.softplus(self.rho)
        if self.posterior_type == 'multiplicative':
            return torch.mul(F.softplus(self.rho), self.mu.pow(2))
        else:
            return torch.mul(F.softplus(self.rho), self.mu.pow(2))

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
    pnorm = torch.sum(p ** 2, dim=1, keepdim=True)
    qnorm = torch.sum(q ** 2, dim=1, keepdim=True)
    norms = (pnorm.expand(pdim, qdim) +
             qnorm.transpose(0, 1).expand(pdim, qdim))
    distances_squared = norms - 2 * p.mm(q.t())
    return torch.sqrt(1e-5 + torch.abs(distances_squared))


def pairwise_distances(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def compute_mmd(x, y, type='inverse', biased=True, space=None, max=False):
    d = x.device

    xs = x.shape[0]
    XX, YY, XY = torch.zeros([xs, xs]).to(d), torch.zeros([xs, xs]).to(d), torch.zeros([xs, xs]).to(d)
    xxd = pdist(x, x) ** 2
    yyd = pdist(y, y) ** 2
    xyd = pdist(x, y) ** 2

    if type == 'rbf':

        if space is None:
            space = [0.5, 1, 2, 4, 8, 16]

        for gamma in space:

            gamma = 1.0 / (2 * gamma ** 2)

            xx = torch.exp(-xxd * gamma)
            yy = torch.exp(-yyd * gamma)
            xy = torch.exp(-xyd * gamma)

            XX += xx
            YY += yy
            XY += xy

    elif type == 'inverse':

        if space is None:
            space = [0.05, 0.2, 0.6, 0.9, 1]

        for a in space:
            a = a ** 2
            xxk = torch.div(1, torch.sqrt(a + xxd))
            yyk = torch.div(1, torch.sqrt(a + yyd))
            xyk = torch.div(1, torch.sqrt(a + xyd))

            XX += xxk
            YY += yyk
            XY += xyk
    else:
        return None

    if biased:
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    else:
        XX = XX.sum() - XX.trace()
        YY = YY.sum() - YY.trace()
        XY = XY.sum()  #- XY.trace()
        mmd = (1 / (xs ** 2)) * XX + (1 / (xs ** 2)) * YY - (2 / (xs * xs)) * XY

    # return torch.sqrt(F.relu(mmd))
    # return torch.sqrt(torch.abs(mmd))
    return F.relu(mmd)
    # return torch.sqrt(torch.abs(mmd))


if __name__ == '__main__':
    torch.manual_seed(10)
    k = 'inverse'
    mmds = []
    for i in range(100):
        x = torch.randn([100, 1])  # * 0.1
        y = torch.randn([100, 1])  # * 0.1 + 10

        a = compute_mmd(x, y, k, biased=True)
        mmds.append(a.item())
    torch.manual_seed(10)

    mmds_u = []
    for i in range(100):
        x = torch.randn([100, 1])  # * 0.1
        y = torch.randn([100, 1])  # * 0.1 + 10

        a = compute_mmd(x, y, k, biased=False)
        mmds_u.append(a.item())

    print(np.mean(mmds), np.mean(mmds_u))
