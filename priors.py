import torch
from torch.distributions import Normal


class Gaussian(object):
    def __init__(self, mu=0, sigma=5):
        self.mu = mu
        self.sigma = sigma
        self.inner_gaussian = Normal(mu, sigma)

    def sample(self, size):
        return self.inner_gaussian.rsample(size)

    def log_prob(self, x):
        return self.inner_gaussian.log_prob(x)


class Laplace(object):
    def __init__(self, mu=0, scale=1):
        self.mu = mu
        self.scale = scale
        self.distribution = torch.distributions.laplace.Laplace(mu, scale)

    def sample(self, size):
        return self.distribution.rsample(size)

    def log_prob(self, x):
        return self.distribution.log_prob(x)


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


class Uniform:
    def __init__(self, a, b):
        self.dist = torch.distributions.uniform.Uniform(a, b)

    def sample(self, size):
        return self.dist.rsample(size)

    def log_prob(self, x):
        return self.dist.log_prob(x.cpu())