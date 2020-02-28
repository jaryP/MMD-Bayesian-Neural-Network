import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from base import Wrapper, get_bayesian_network, cross_entropy_loss, log_gaussian_loss, Network
from priors import Gaussian
from bayesian_layers import BayesianCNNLayer, BayesianLinearLayer


class BBB(Network):

    def __init__(self, sample, classes, topology=None, prior=None, mu_init=None, rho_init=None,
                 local_trick=False, regression=False, posterior_type='weights', **kwargs):
        super().__init__(classes=classes, regression=regression)

        if topology is None:
            topology = [400, 400]

        if prior is None:
            prior = Gaussian(0, 10)

        self.calculate_kl = True

        self._prior = prior
        self.features = get_bayesian_network(topology, sample, classes,
                                             mu_init, rho_init, prior, 'kl', local_trick, bias=True,
                                             posterior_type=posterior_type)

    def _forward(self, x):

        tot_prior = 0
        tot_post = 0

        for j, i in enumerate(self.features):
            if not isinstance(i, (BayesianLinearLayer, BayesianCNNLayer)):
                x = i(x)
            else:
                x, prior, post = i(x,  self.calculate_kl)
                tot_post += post
                tot_prior += prior

        return x, tot_prior, tot_post

    def forward(self, x, samples=1):
        o = []
        log_priors = []
        log_posts = []

        for i in range(samples):
            op, prior, post = self._forward(x)
            o.append(op)
            log_priors.append(prior)
            log_posts.append(post)

        o = torch.stack(o)
        log_priors = torch.stack(log_priors)
        log_posts = torch.stack(log_posts)

        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        return o, log_prior, log_post

    def eval_forward(self, x, samples=1):
        o, _, _ = self(x, samples=samples)
        return o


class Trainer(Wrapper):
    def __init__(self, model: nn.Module, train_data, test_data, optimizer, **kwargs):
        super().__init__(model, train_data, test_data, optimizer)

        self.regression = model.regression
        if model.regression:
            self.loss = log_gaussian_loss(model.classes)
        else:
            self.loss = cross_entropy_loss('sum')

    def train_epoch(self, samples=1, **kwargs):
        losses = []

        M = len(self.train_data)
        a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
        b = 2 ** M - 1

        pi = a / b

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=False, leave=False)
        progress_bar.set_postfix(ce_loss='', kl_loss='')

        train_true = []
        train_pred = []
        self.model.calculate_kl = True

        for batch, (x, y) in progress_bar:
            train_true.extend(y.tolist())
            y = y.to(self.device)
            x = x.to(self.device)

            self.optimizer.zero_grad()

            out, prior, post = self.model(x, samples=samples)

            logloss = (post - prior) * pi[batch] #/ x.shape[0]

            if pi[batch] == 0:
                self.model.calculate_kl = False

            if self.regression:
                out = out.mean(0)
                logloss = logloss/x.shape[0]

                if self.model.classes == 1:
                    noise = self.model.noise.exp()
                    x = out
                    loss = self.loss_function(x, y, noise)
                else:
                    loss = self.loss_function(out[:, :1], y, out[:, 1:].exp())
                loss = loss/x.shape[0]
            else:
                loss = self.loss_function(out, y)
                out = torch.softmax(out, -1).mean(0)
                out = out.argmax(dim=-1)

            progress_bar.set_postfix(ce_loss=loss.item(), kl_loss=logloss.item())
            loss += logloss

            losses.append(loss.item())
            loss.backward()

            self.optimizer.step()

            train_pred.extend(out.tolist())

        return losses, (train_true, train_pred)

    def train_step(self, train_samples=1, test_samples=1, **kwargs):
        losses, train_res = self.train_epoch(samples=train_samples)
        test_res = self.test_evaluation(samples=test_samples)
        return losses, train_res, test_res

