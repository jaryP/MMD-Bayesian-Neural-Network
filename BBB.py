import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm

from base import BayesianLinearLayer
import torch


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
                BayesianLinearLayer(in_size=prev, out_size=i, mu_init=mu_init, divergence='kl',
                                    rho_init=rho_init, prior=self._prior))
            prev = i

        self.classificator = nn.ModuleList(
            [BayesianLinearLayer(in_size=prev, out_size=classes, mu_init=mu_init, rho_init=rho_init, divergence='kl',
                                 prior=self._prior)])

    def forward(self, x):

        tot_prior = 0
        tot_post = 0
        for i in self.features:
            x, prior, post = i(x)
            tot_post += post
            tot_prior += prior
            x = torch.relu(x)

        for i in self.classificator:
            x, prior, post = i(x)
            tot_post += post
            tot_prior += prior

        return x, tot_prior, tot_post

    def sample_forward(self, x, samples=1):
        o = []
        log_priors = []
        log_posts = []

        for i in range(samples):
            op, prior, post = self(x)
            o.append(op)
            log_priors.append(prior)
            log_posts.append(post)

        o = torch.stack(o)

        log_priors = torch.stack(log_priors)
        log_posts = torch.stack(log_posts)

        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        return o, log_prior, log_post


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

