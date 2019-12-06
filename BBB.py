from itertools import chain

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm

from base import BayesianLinearLayer, Gaussian, Network, Wrapper
import torch


class BBB(Network):

    def __init__(self, input_size, classes, topology=None, prior=None, mu_init=None, rho_init=None,
                 local_trick=False, **kwargs):
        super().__init__()

        if mu_init is None:
            mu_init = (-0.6, 0.6)

        if rho_init is None:
            rho_init = -6

        if topology is None:
            topology = [400, 400]

        if prior is None:
            prior = Gaussian(0, 10)

        self.features = nn.ModuleList()
        self._prior = prior

        prev = input_size

        for i in topology:
            self.features.append(
                BayesianLinearLayer(in_size=prev, out_size=i, mu_init=mu_init, divergence='kl',
                                    rho_init=rho_init, prior=self._prior, local_rep_trick=local_trick))
            prev = i

        self.classificator = nn.ModuleList(
            [BayesianLinearLayer(in_size=prev, out_size=classes, mu_init=mu_init, rho_init=rho_init, divergence='kl',
                                 prior=self._prior, local_rep_trick=local_trick)])

    def _forward(self, x):

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

    def layers(self):
        return chain(self.features, self.classificator)

    def eval_forward(self, x, samples=1):
        o, _, _ = self(x, samples=samples)
        return o


class Trainer(Wrapper):
    def __init__(self, model: nn.Module, train_data, test_data, optimizer):
        super().__init__(model, train_data, test_data, optimizer)

    def train_epoch(self, samples=1, **kwargs):
        losses = []

        M = len(self.train_data)
        a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
        b = 2 ** M - 1

        pi = a / b

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=True)
        # progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

        train_true = []
        train_pred = []

        for batch, (x_train, y_train) in progress_bar:
            train_true.extend(y_train.tolist())

            self.optimizer.zero_grad()

            out, prior, post = self.model(x_train.to(self.device), samples=samples)
            out = out.mean(0)
            logloss = (- post - prior) * pi[batch]  # /x_train.shape[0]

            ce = F.nll_loss(F.log_softmax(out, -1), y_train.to(self.device))
            loss = ce + logloss
            losses.append(loss.item())
            loss.backward()

            self.optimizer.step()

            max_class = F.log_softmax(out, -1).argmax(dim=-1)
            train_pred.extend(max_class.tolist())
            # progress_bar.set_postfix(ce_loss=loss.item())

        return losses, (train_true, train_pred)

    def test_evaluation(self, samples, **kwargs):

        test_pred = []
        test_true = []

        self.model.eval()
        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(self.test_data):
                test_true.extend(y_test.tolist())

                out = self.model.eval_forward(x_test.to(self.device), samples=samples)
                out = out.mean(0)
                out = out.argmax(dim=-1)
                test_pred.extend(out.tolist())

        return test_true, test_pred

    def train_step(self, train_samples=1, test_samples=1, **kwargs):
        losses, train_res = self.train_epoch(samples=train_samples)
        test_res = self.test_evaluation(samples=test_samples)
        return losses, train_res, test_res

    def snr_test(self, percentiles: list):
        return None


# def epoch(model, optimizer, train_dataset, test_dataset, train_samples, test_samples, device, weights, **kwargs):
#     losses = []
#
#     # mmd_w = weights.get('mmd', 1)
#
#     M = len(train_dataset)
#     a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
#     b = 2 ** M - 1
#
#     pi = a / b
#
#     model.train()
#     progress_bar = tqdm(enumerate(train_dataset), total=len(train_dataset), disable=True)
#     # progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')
#
#     train_true = []
#     train_pred = []
#
#     for batch, (x_train, y_train) in progress_bar:
#         train_true.extend(y_train.tolist())
#
#         optimizer.zero_grad()
#
#         out, prior, post = model.sample_forward(x_train.to(device), samples=train_samples)
#         out = out.mean(0)
#         logloss = (- post - prior) * pi[batch] # /x_train.shape[0]
#         # mmd = (mmd * mmd_w)
#
#         max_class = F.log_softmax(out, -1).argmax(dim=-1)
#         train_pred.extend(max_class.tolist())
#
#         ce = F.nll_loss(F.log_softmax(out, -1), y_train.to(device))
#         loss = ce + logloss
#         losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
#
#         progress_bar.set_postfix(log_loss=logloss.item(), ce_loss=ce.item())
#
#     test_pred = []
#     test_true = []
#
#     model.eval()
#     with torch.no_grad():
#         for i, (x_test, y_test) in enumerate(test_dataset):
#             test_true.extend(y_test.tolist())
#
#             out, _, _ = model.sample_forward(x_test.to(device), samples=test_samples)
#             out = out.mean(0)
#             out = out.argmax(dim=-1)
#             test_pred.extend(out.tolist())
#
#     return losses, (train_true, train_pred), (test_true, test_pred)


if __name__ == '__main__':
    from itertools import chain

    import torch
    from sklearn import metrics
    from torch import nn
    from torchvision.transforms import transforms
    from torchvision import datasets
    from tqdm import tqdm

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.Normalize((0,), (1,)),
        torch.flatten
    ])

    train_split = datasets.MNIST('./Datasets/MNIST', train=True, download=True,
                                 transform=image_transform)

    train_loader = torch.utils.data.DataLoader(train_split, batch_size=100, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Datasets/MNIST', train=False, transform=image_transform), batch_size=1000,
        shuffle=False)

    input_size = 784
    classes = 10

    ann = BBB(784, 10, local_trick=False)
    ann.cuda()
    trainer = Trainer(ann, train_loader, test_loader, torch.optim.Adam(ann.parameters(), 1e-3))

    for i in range(10):
       a, _, (test_true, test_pred) = trainer.train_step(test_samples=2, train_samples=2)
       f1 = metrics.f1_score(test_true, test_pred, average='micro')

       print(f1)