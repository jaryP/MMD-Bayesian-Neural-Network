from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from base import Gaussian, Network, Wrapper, get_bayesian_network, cross_entropy_loss, log_gaussian_loss
from bayesian_utils import BayesianCNNLayer, BayesianLinearLayer


class BBB(Network):

    def __init__(self, sample, classes, topology=None, prior=None, mu_init=None, rho_init=None,
                 local_trick=False, regression=False, **kwargs):
        super().__init__(classes=classes, regression=regression)

        # if mu_init is None:
        #     mu_init = (-0.6, 0.6)
        #
        # if rho_init is None:
        #     rho_init = -6

        if topology is None:
            topology = [400, 400]

        if prior is None:
            prior = Gaussian(0, 10)

        self.calculate_kl = True

        self._prior = prior
        self.features = get_bayesian_network(topology, sample, classes,
                                             mu_init, rho_init, prior, 'kl', local_trick, bias=True)

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

                # if j < len(self.features)-1:
                #     x = torch.relu(x)

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
        progress_bar.set_postfix(loss='')

        train_true = []
        train_pred = []
        self.model.calculate_kl = True

        for batch, (x, y) in progress_bar:
            train_true.extend(y.tolist())
            y = y.to(self.device)
            x = x.to(self.device)

            self.optimizer.zero_grad()

            out, prior, post = self.model(x, samples=samples)
            out = out.mean(0)

            logloss = (post - prior) * pi[batch] #/ x.shape[0]

            if logloss == 0:
                self.model.calculate_kl = False

            if self.regression:
                if self.model.classes == 1:
                    noise = self.model.noise.exp()
                    x = out
                    loss = self.loss_function(x, y, noise)
                else:
                    loss = self.loss_function(out[:, :1], y, out[:, 1:].exp())#/x.shape[0]
            else:
                loss = self.loss_function(out, y)
                out = out.argmax(dim=-1)

            # ce = F.cross_entropy(out, y.to(self.device), reduction='sum')

            loss += logloss
            progress_bar.set_postfix(loss=loss.item())

            losses.append(loss.item())
            loss.backward()

            self.optimizer.step()

            # max_class = F.log_softmax(out, -1).argmax(dim=-1)
            train_pred.extend(out.tolist())

        return losses, (train_true, train_pred)

    def test_evaluation(self, samples, **kwargs):

        test_pred = []
        test_true = []

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_data):
                test_true.extend(y.tolist())

                out = self.model.eval_forward(x.to(self.device), samples=samples)
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

    # def rotation_test(self, samples=1):
    #     ts_copy = copy(self.test_data.dataset.transform)
    #
    #     HS = []
    #     DIFF = []
    #     scores = []
    #     self.model.eval()
    #
    #     for angle in self.rotations:
    #         ts = T.Compose([AngleRotation(angle), ts_copy])
    #         self.test_data.dataset.transform = ts
    #
    #         H = []
    #         pred_label = []
    #         true_label = []
    #
    #         diff = []
    #
    #         outs = []
    #         self.model.eval()
    #         with torch.no_grad():
    #             for i, (x, y) in enumerate(self.test_data):
    #                 true_label.extend(y.tolist())
    #                 y = y.to(self.device)
    #                 out = self.model.eval_forward(x.to(self.device), samples=samples)
    #
    #                 outs.append(out.cpu().numpy())
    #
    #                 out_m = out.mean(0)
    #                 pred_label.extend(out_m.argmax(dim=-1).tolist())
    #
    #                 top_score, top_label = torch.topk(F.softmax(out.mean(0), -1), 2)
    #
    #                 H.extend(top_score[:, 0].tolist())
    #                 diff.extend(((top_score[:, 0] - top_score[:, 1]) ** 2).tolist())
    #
    #         p_hat = np.asarray(H)
    #
    #         mean_diff = np.mean(diff)
    #
    #         epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    #         aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    #
    #         entropy = aleatoric + epistemic
    #
    #         # HS.append((np.mean(correct_h), np.mean(incorrect_h)))
    #
    #         # HS.append(np.mean(H))
    #
    #         HS.append(entropy)
    #         DIFF.append(mean_diff)
    #         scores.append(metrics.f1_score(true_label, pred_label, average='micro'))
    #
    #     self.test_data.dataset.transform = ts_copy
    #     return HS, DIFF, scores

    # def attack_test(self, samples=1):
    #     HS = []
    #     scores = []
    #
    #     # M = len(self.test_data)
    #     # a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
    #     # b = 2 ** M - 1
    #     #
    #     # pi = a / b
    #     self.model.eval()
    #
    #     for eps in self.epsilons:
    #
    #         H = []
    #         pred_label = []
    #         true_label = []
    #
    #         # self.model.eval()
    #         for i, (x, y) in enumerate(self.test_data):
    #             self.model.zero_grad()
    #
    #             true_label.extend(y.tolist())
    #             self.model.zero_grad()
    #
    #             x = x.to(self.device)
    #             y = y.to(self.device)
    #             x.requires_grad = True
    #
    #             out, prior, post = self.model(x, samples=samples)
    #             out = out.mean(0)
    #             pred = (out.argmax(dim=-1) == y).tolist()
    #
    #             # logloss = (post - prior) * pi[i]  # */x_train.shape[0]   #
    #
    #             ce = F.cross_entropy(out, y, reduction='mean')
    #             # loss = ce + logloss
    #
    #             ce.backward()
    #
    #             perturbed_data = fgsm_attack(x, eps)
    #             out = self.model.eval_forward(perturbed_data, samples=samples)
    #             out = out.mean(0)
    #
    #             a = compute_entropy(F.softmax(out, -1), True).tolist()
    #             pred_label.extend(out.argmax(dim=-1).tolist())
    #             H.extend([a[i] for i in range(len(a)) if pred[i] == 0])
    #
    #             # H.extend(compute_entropy(F.softmax(out, -1)).tolist())
    #
    #         scores.append(metrics.f1_score(true_label, pred_label, average='micro'))
    #         HS.append(np.mean(H))
    #
    #     return HS, scores
