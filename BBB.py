import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

from base import BayesianParameters, Gaussian, ScaledMixtureGaussian
import torch


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
        self.log_prior = None
        self.log_posterior = None

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

        return o


class BBB(nn.Module):

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

        self.features = nn.ModuleList()
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

    def forward(self, x, sample=1, task=None):

        for i in self.features:
            x = torch.relu(i(x))

        for i in self.classificator:
            x = i(x)

        return x, self.prior(), self.posterior()

    def sample_forward(self, x, samples=1, task=None):
        o = []
        log_priors = []
        log_posts = []

        for i in range(samples):
            op, prior, post = self(x, task=task)
            o.append(op)
            log_priors.append(prior)
            log_posts.append(post)

        o = torch.stack(o)

        log_priors = torch.stack(log_priors)
        log_posts = torch.stack(log_posts)

        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        return o, log_prior, log_post


# def epoch(model, optimizer, train_dataset, test_dataset, train_samples, test_samples, device, **kwargs):
#     losses = []
#
#     M = len(train_dataset)
#     a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
#     b = 2 ** M - 1
#
#     pi = a / b
#
#     model.train()
#     for batch, (x_train, y_train) in enumerate(train_dataset):
#         optimizer.zero_grad()
#
#         out, prior, post = model.sample_forward(x_train.to(device), samples=train_samples)
#         out = out.mean(0)
#         loss = (- post - prior) * pi[batch]
#         loss += F.cross_entropy(out, y_train.to(device))
#         losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
#
#     preds = []
#     true_val = []
#
#     model.eval()
#     with torch.no_grad():
#         for i, (x_test, y_test) in enumerate(test_dataset):
#             true_val.extend(y_test.tolist())
#
#             out, _, _ = model.sample_forward(x_test.to(device), samples=test_samples)
#             out = out.mean(0)
#             out = out.argmax(dim=-1)
#             preds.extend(out.tolist())
#
#     return losses, preds, true_val


def epoch(model, optimizer, train_dataset, test_dataset, train_samples, test_samples, device, weights, **kwargs):
    losses = []

    # mmd_w = weights.get('mmd', 1)

    M = len(train_dataset)
    a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
    b = 2 ** M - 1

    pi = a / b

    model.train()
    progress_bar = tqdm(enumerate(train_dataset), total=len(train_dataset), disable=True)
    # progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

    train_true = []
    train_pred = []

    for batch, (x_train, y_train) in progress_bar:
        train_true.extend(y_train.tolist())

        optimizer.zero_grad()

        out, prior, post = model.sample_forward(x_train.to(device), samples=train_samples)
        out = out.mean(0)
        logloss = (- post - prior) * pi[batch]
        # mmd = (mmd * mmd_w)

        max_class = F.log_softmax(out, -1).argmax(dim=-1)
        train_pred.extend(max_class.tolist())

        ce = F.nll_loss(F.log_softmax(out, -1), y_train.to(device))
        loss = ce + logloss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(log_loss=logloss.item(), ce_loss=ce.item())

    test_pred = []
    test_true = []

    model.eval()
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(test_dataset):
            test_true.extend(y_test.tolist())

            out, _, _ = model.sample_forward(x_test.to(device), samples=test_samples)
            out = out.mean(0)
            out = out.argmax(dim=-1)
            test_pred.extend(out.tolist())

    return losses, (train_true, train_pred), (test_true, test_pred)


# def trainer(epochs, model, train_dataset, test_dataset):
#     for epoch in range(epochs):
#         print(epoch)
#         for batch, (x_train, y_train) in enumerate(train_dataset):
#             optimizer.zero_grad()
#
#             out, prior, post = model.sample_forward(x_train, samples=1)
#             out = out.mean(0)
#
#             loss = (- post - prior) * pi[batch]
#
#             loss += F.cross_entropy(out, y_train)
#
#             loss.backward()
#             optimizer.step()
#
#         # learning_rate /= 1.1
#         test_losses, test_accs = [], []
#         for i, (x_test, y_test) in enumerate(test_dataset):
#             optimizer.zero_grad()
#             out, prior, post = model.sample_forward(x_test, samples=5)
#             out = out.mean(0)
#             # loss = F.cross_entropy(pred, y_test)
#             acc = (out.argmax(dim=-1) == y_test).to(torch.float32).mean()
#             # test_losses.append(loss.item())
#             test_accs.append(acc.mean().item())
#
#         print('Accuracy: {}'.format(np.mean(test_accs)))
#     print('Finished Training')
#
# if __name__ == '__main__':
#
#     import numpy as np
#     import torch
#     from torch import optim, nn
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     n_epochs = 3
#     batch_size_train = 128
#     batch_size_test = 1000
#     learning_rate = 0.001
#     momentum = 0.5
#     log_interval = 10
#     input_size = 784  # The image size = 28 x 28 = 784
#
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            # transforms.Normalize((0,), (1,)),
#                            torch.flatten
#                        ])), batch_size=batch_size_train, shuffle=True)
#
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('data', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Normalize((0,), (1,)),
#             torch.flatten
#         ])), batch_size=batch_size_test, shuffle=True)
#
#     model = BBB(input_size=input_size, classes=10,
#                 device=device, prior=ScaledMixtureGaussian(0.5, np.exp(0), np.exp(-6)))
#     epochs = 10
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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
#             out, prior, post = model.sample_forward(x_train, samples=1)
#             out = out.mean(0)
#
#             loss = (- post - prior) * pi[batch]
#
#             loss += F.cross_entropy(out, y_train)
#
#             loss.backward()
#             optimizer.step()
#
#         # learning_rate /= 1.1
#         test_losses, test_accs = [], []
#         for i, (x_test, y_test) in enumerate(test_loader):
#             optimizer.zero_grad()
#             out, prior, post = model.sample_forward(x_test, samples=5)
#             out = out.mean(0)
#             # loss = F.cross_entropy(pred, y_test)
#             acc = (out.argmax(dim=-1) == y_test).to(torch.float32).mean()
#             # test_losses.append(loss.item())
#             test_accs.append(acc.mean().item())
#
#         print('Accuracy: {}'.format(np.mean(test_accs)))
#     print('Finished Training')
