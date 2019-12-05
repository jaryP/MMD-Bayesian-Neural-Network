import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision import datasets, transforms
from tqdm import tqdm

from base import BayesianParameters


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    # x = x.unsqueeze(1).expand(n, m, d)
    # y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(torch.sub(x.unsqueeze(1).expand(n, m, d), y.unsqueeze(0).expand(n, m, d)), 2).sum(2)


def pairwise_distances(x, y):

    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def cdist2(x, y):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.pow(2).sum(dim=-1)
    y_sq_norm = y.pow(2).sum(dim=-1)
    x_dot_y = x @ y.t()
    sq_dist = x_sq_norm.unsqueeze(dim=1) + y_sq_norm.unsqueeze(dim=0) - 2*x_dot_y
    # For numerical issues
    sq_dist.clamp_(min=0.0)
    return torch.sqrt(sq_dist)


def compute_kernel(x, y):
    dim = x.size(1)
    # d = pairwise_distances(x, y)
    # print(d.size())
    # d = torch.exp(- torch.mul(torch.cdist(x, y).mean(1), 1/float(dim))).mean()
    d = torch.exp(- torch.mul(pairwise_distances(x, y).mean(1), 1/float(dim))).mean()
    # d = torch.exp(- torch.mul(pairwise_distances1(x, y).mean(1), 1/float(dim))).mean()

    # d2 = cdist2(x, y)
    # d1 = torch.norm(x[:, None] - y, dim=2, p=2)
    # d = (- torch.norm(x[:, None] - y, dim=2, p=2).mean(1) / float(dim)).exp().mean()
    # d = (- row_pairwise_distances(x, y) / float(dim)).exp().mean().to(x.device)
    # d1 = x**2 + y**2 - 2*torch.matmul(x.t(), y)
    # d1 = torch.norm(x - y, 2)
    return d


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return x_kernel + y_kernel - 2 * xy_kernel


class BayesianLinearLayer(nn.Module):
    def __init__(self, in_size, out_size, mu_init=None, rho_init=None, use_bias=True, prior=None):

        super().__init__()

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
        self.mmd_w = None
        self.mmd_b = None

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
        w = self.w.weights

        b = None
        mmd_w = torch.tensor(0).float()
        mmd_b = torch.tensor(0).float()

        # if self.b is not None:
        #     b = self.b.weights.unsqueeze(0)
        #     mmd_b = compute_mmd(b, self.prior_b.rsample(b.size()).to(w.device))

        if self.training:
            mmd_w = compute_mmd(w, self.prior_w.sample(w.size()).to(w.device))

            if self.b is not None:
                b = self.b.weights.unsqueeze(0)
                mmd_b = compute_mmd(b, self.prior_b.sample(b.size()).to(b.device))

        o = F.linear(x, w, b)

        self.w_w = w
        self.b_w = b

        return o, mmd_w + mmd_b


class BMMD(nn.Module):

    def __init__(self, input_size, classes, topology=None, prior=None, mu_init=None, rho_init=None, **kwargs):
        super().__init__()

        if mu_init is None:
            mu_init = (-0.6, 0.6)

        if rho_init is None:
            rho_init = -6

        if topology is None:
            topology = [400, 400]

        if prior is None:
            prior = Normal(0, 10)

        self.features = torch.nn.ModuleList()
        self._prior = prior

        prev = input_size

        for i in topology:
            self.features.append(
                BayesianLinearLayer(in_size=prev, out_size=i, mu_init=mu_init,
                                    rho_init=rho_init, prior=self._prior))
            prev = i

        self.classificator = nn.ModuleList(
            [BayesianLinearLayer(in_size=prev, out_size=classes, mu_init=mu_init, rho_init=rho_init,
                                 prior=self._prior)])

    def prior(self):
        p = 0
        for i in self.features:
            p += i.log_prior

        for i in self.classificator:
            p += i.log_prior

        return p

    def posterior(self):
        p = 0
        for i in self.features:
            p += i.log_posterior
        for i in self.classificator:
            p += i.log_posterior
        return p

    def mmd(self):
        mmd = 0
        for i in self.features:
            mmd += i.mmd()
        for i in self.classificator:
            mmd += i.mmd()
        return mmd

    def forward(self, x, sample=1, task=None):

        mmd = 0
        for j, i in enumerate(self.features):
            r, m = i(x)
            mmd += m
            x = torch.relu(r)

        x, m = self.classificator[0](x)

        mmd += m

        return x, mmd

    def sample_forward(self, x, samples=1, task=None):
        o = []
        mmds = []

        for i in range(samples):
            op, mmd = self(x, task=task)
            o.append(op)
            mmds.append(mmd)

        o = torch.stack(o)

        mmds = torch.stack(mmds).mean()

        return o, mmds


def epoch(model, optimizer, train_dataset, test_dataset, train_samples, test_samples, device, weights, **kwargs):
    losses = []

    mmd_w = weights.get('mmd', 1)

    model.train()
    progress_bar = tqdm(enumerate(train_dataset), total=len(train_dataset), disable=True)
    # progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

    train_true = []
    train_pred = []

    for batch, (x_train, y_train) in progress_bar:
        train_true.extend(y_train.tolist())

        optimizer.zero_grad()

        out, mmd = model.sample_forward(x_train.to(device), samples=train_samples)
        out = out.mean(0)
        mmd = (mmd * mmd_w)

        max_class = F.log_softmax(out, -1).argmax(dim=-1)
        train_pred.extend(max_class.tolist())

        ce = F.nll_loss(F.log_softmax(out, -1), y_train.to(device))
        loss = ce + mmd
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(mmd_loss=mmd.item(), ce_loss=ce.item())

    test_pred = []
    test_true = []

    model.eval()
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(test_dataset):
            test_true.extend(y_test.tolist())

            out, mmd = model.sample_forward(x_test.to(device), samples=test_samples)
            out = out.mean(0)
            out = out.argmax(dim=-1)
            test_pred.extend(out.tolist())

    return losses, (train_true, train_pred), (test_true, test_pred)


# if __name__ == '__main__':
#
#     import numpy as np
#     import torch
#     from torch import optim, nn
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(device)
#
#     batch_size_train = 128
#     batch_size_test = 128
#     learning_rate = 0.001
#     input_size = 784  # The image size = 28 x 28 = 784
#     epochs = 100
#
#     image_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0,), (1,)),
#         torch.flatten
#     ])
#
#     train_split = datasets.MNIST('data', train=True, download=True,
#                                  transform=image_transform)
#
#     train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size_train, shuffle=True)
#
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('data', train=False, transform=image_transform), batch_size=batch_size_test, shuffle=True)
#
#     prior = Normal(0, 1)
#     model = BMMD(input_size=input_size, classes=10, mu_init=(-1, 1), rho_init=-3,
#                  device=device, prior=prior)  # , prior=ScaledMixtureGaussian(0.5, np.exp(0), np.exp(-6)))
#
#     model.to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     M = len(train_loader)
#     a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
#     b = np.asarray([2 ** M - 1 for i in range(M + 1)])
#
#     pi = a / b
#
#     for epoch in range(epochs):
#         print(epoch)
#         for batch, (x_train, y_train) in enumerate(train_loader):
#             optimizer.zero_grad()
#
#             out, mmd = model.sample_forward(x_train.to(device), samples=2)
#             out = out.mean(0)
#             # loss = (- post - prior) * pi[batch]
#             loss = (mmd * 1) # * pi[batch]
#             # print(loss, end=', ')
#             loss += F.cross_entropy(out, y_train.to(device))
#             # print(loss)
#             loss.backward()
#             optimizer.step()
#
#         # learning_rate /= 1.1
#         test_losses, test_accs = [], []
#         with torch.no_grad():
#             for i, (x_test, y_test) in enumerate(test_loader):
#                 optimizer.zero_grad()
#                 out, mmd = model.sample_forward(x_test.to(device), samples=5)
#                 out = out.mean(0)
#                 # loss = F.cross_entropy(pred, y_test)
#                 acc = (out.argmax(dim=-1) == y_test.to(device)).to(torch.float32).mean()
#                 # test_losses.append(loss.item())
#                 test_accs.append(acc.mean().item())
#
#             print('Accuracy: {}'.format(np.mean(test_accs)))
#     print('Finished Training')
