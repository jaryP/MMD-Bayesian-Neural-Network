import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm
import torch.nn.functional as F


def pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def compute_kernel(x, y):
    dim = x.size(1)
    # d = torch.exp(- torch.mul(torch.cdist(x, y).mean(1), 1/float(dim))).mean()
    d = torch.exp(- torch.mul(pairwise_distances(x, y).mean(1), 1/float(dim))).mean()
    return d


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return x_kernel + y_kernel - 2 * xy_kernel


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
        return self.mu + torch.log(1 + torch.exp(self.rho)) * Normal(0, 1).sample(self.mu.shape).to(self.mu.device)

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


class BayesianLinearLayer(nn.Module):
    def __init__(self, in_size, out_size, divergence, mu_init=None, rho_init=None, use_bias=True, prior=None):

        super().__init__()

        divergence = divergence.lower()
        if divergence not in ['mmd', 'kl']:
            raise ValueError('type parameter should be mmd or bbb.')

        self.divergence = divergence

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

    def _mmd_forward(self, x):
        w = self.w.weights

        b = None
        mmd_w = torch.tensor(0).float()
        mmd_b = torch.tensor(0).float()

        if self.training:
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device))

            if self.b is not None:
                b = self.b.weights.unsqueeze(0)
                mmd_b = compute_mmd(b, self.prior_b.sample(b.size()).to(b.device))

        o = F.linear(x, w, b)

        self.w_w = w
        self.b_w = b

        return o, mmd_w + mmd_b

    def _kl_forward(self, x):
        w = self.w.weights
        log_post = self.w.posterior_log_prob(w).sum()
        log_prior = self.prior_w.log_prob(w).sum()
        b = None

        if self.b is not None:
            b = self.b.weights

            log_post += self.b.posterior_log_prob(b).sum()
            log_prior += self.prior_b.log_prob(b).sum()

        o = F.linear(x, w, b)

        self.log_prior = log_prior
        self.log_posterior = log_post
        self.w_w = w
        self.b_w = b

        return o, log_prior, log_post


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
        # w = self.w.weights
        # log_post = self.w.posterior_log_prob(w).sum()
        # log_prior = self.prior_w.log_prob(w).sum()
        # b = None
        #
        # if self.b is not None:
        #     b = self.b.weights
        #
        #     log_post += self.b.posterior_log_prob(b).sum()
        #     log_prior += self.prior_b.log_prob(b).sum()
        #
        # o = F.linear(x, w, b)
        #
        # self.log_prior = log_prior
        # self.log_posterior = log_post
        # self.w_w = w
        # self.b_w = b
        #
        # return o


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

    def prob(self, x):
        return self.inner_gaussian.prob(x)


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
        return self.pi * self.gaussian1.sample(size) + (1-self.pi)*self.gaussian2.sample(size)

    def log_prob(self, x):
        return self.pi * self.gaussian1.log_prob(x) + (1-self.pi)*self.gaussian2.log_prob(x)

    def prob(self, x):
        return self.pi * self.gaussian1.prob(x) + (1-self.pi)*self.gaussian2.prob(x)


class ANN(nn.Module):

    def __init__(self, input_size, classes, topology=None, prior=None, **kwargs):
        super().__init__()

        if topology is None:
            topology = [400, 400]

        if prior is None:
            prior = Normal(0, 10)

        self.features = torch.nn.ModuleList()
        self._prior = prior

        prev = input_size

        for i in topology:
            self.features.append(
                torch.nn.Linear(prev, i))
            prev = i

        self.classificator = nn.ModuleList([torch.nn.Linear(prev, classes)])

    def forward(self, x, sample=1, task=None):

        for j, i in enumerate(self.features):
            r = i(x)
            x = torch.relu(r)

        x = self.classificator[0](x)

        return x

    # def sample_forward(self, x, samples=1, task=None):
    #     o = []
    #     mmds = []
    #
    #     for i in range(samples):
    #         op, mmd = self(x, task=task)
    #         o.append(op)
    #         mmds.append(mmd)
    #
    #     o = torch.stack(o)
    #
    #     mmds = torch.stack(mmds).mean()
    #
    #     return o, mmds


def epoch(model, optimizer, train_dataset, test_dataset, device, **kwargs):
    losses = []

    model.train()
    progress_bar = tqdm(enumerate(train_dataset), total=len(train_dataset), disable=True)
    # progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

    train_true = []
    train_pred = []

    for batch, (x_train, y_train) in progress_bar:
        train_true.extend(y_train.tolist())

        optimizer.zero_grad()

        out = model(x_train.to(device))

        max_class = F.log_softmax(out, -1).argmax(dim=-1)
        train_pred.extend(max_class.tolist())

        loss = F.nll_loss(F.log_softmax(out, -1), y_train.to(device))

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(ce_loss=loss.item())

    test_pred = []
    test_true = []

    model.eval()
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(test_dataset):
            test_true.extend(y_test.tolist())

            out = model(x_test.to(device))
            out = out.argmax(dim=-1)
            test_pred.extend(out.tolist())

    return losses, (train_true, train_pred), (test_true, test_pred)
