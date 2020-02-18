import numpy as np

from torch import nn as nn
from torch.distributions import Normal
from torch.nn import functional as F, init
import torch


class BayesianParameters(nn.Module):
    def __init__(self, size, mu_initialization=None, rho_initialization=None, posterior_type='weights', is_bias=False):
        super().__init__()
        self.posterior_type = posterior_type
        self.mask = None

        if mu_initialization is None:
            t = torch.empty(size)
            if is_bias:
                bound = 1 / np.sqrt(size)
                init.uniform_(t, -bound, bound)
            else:
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
            percentile = np.percentile(snr.cpu(), self.mask * 100)
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

    def posterior_distribution(self):
        return Normal(self.mu.data.clone(), torch.log(1 + torch.exp(self.rho)).clone())

    def posterior_log_prob(self, w):
        return self.posterior_distribution().log_prob(w)

    def forward(self, input, sample=1):
        pass


def b_drop(x, p=0.5):
    return F.dropout(x, p=p, training=True, inplace=False)


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
        XY = XY.sum()  # - XY.trace()
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
