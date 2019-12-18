from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from base import Network, Gaussian, Wrapper, get_bayesian_network
from bayesian_utils import BayesianCNNLayer, BayesianLinearLayer


# def euclidean_dist( x, y):
#     # x: N x D
#     # y: M x D
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     assert d == y.size(1)
#
#     # x = x.unsqueeze(1).expand(n, m, d)
#     # y = y.unsqueeze(0).expand(n, m, d)
#
#     return torch.pow(torch.sub(x.unsqueeze(1).expand(n, m, d), y.unsqueeze(0).expand(n, m, d)), 2).sum(2)
#
#
# def pairwise_distances(x, y):
#
#     x_norm = (x**2).sum(1).view(-1, 1)
#     y_norm = (y ** 2).sum(1).view(1, -1)
#
#     dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
#     return torch.clamp(dist, 0.0, np.inf)
#
#
# def cdist2(x, y):
#     # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
#     x_sq_norm = x.pow(2).sum(dim=-1)
#     y_sq_norm = y.pow(2).sum(dim=-1)
#     x_dot_y = x @ y.t()
#     sq_dist = x_sq_norm.unsqueeze(dim=1) + y_sq_norm.unsqueeze(dim=0) - 2*x_dot_y
#     # For numerical issues
#     sq_dist.clamp_(min=0.0)
#     return torch.sqrt(sq_dist)
#
#
# def compute_kernel(x, y):
#     dim = x.size(1)
#     # d = pairwise_distances(x, y)
#     # print(d.size())
#     # d = torch.exp(- torch.mul(torch.cdist(x, y).mean(1), 1/float(dim))).mean()
#     d = torch.exp(- torch.mul(pairwise_distances(x, y).mean(1), 1/float(dim))).mean()
#     # d = torch.exp(- torch.mul(pairwise_distances1(x, y).mean(1), 1/float(dim))).mean()
#
#     # d2 = cdist2(x, y)
#     # d1 = torch.norm(x[:, None] - y, dim=2, p=2)
#     # d = (- torch.norm(x[:, None] - y, dim=2, p=2).mean(1) / float(dim)).exp().mean()
#     # d = (- row_pairwise_distances(x, y) / float(dim)).exp().mean().to(x.device)
#     # d1 = x**2 + y**2 - 2*torch.matmul(x.t(), y)
#     # d1 = torch.norm(x - y, 2)
#     return d
#
#
# def compute_mmd(x, y):
#     x_kernel = compute_kernel(x, x)
#     y_kernel = compute_kernel(y, y)
#     xy_kernel = compute_kernel(x, y)
#     return x_kernel + y_kernel - 2 * xy_kernel


class BMMD(Network):

    def __init__(self, sample, classes, topology=None, prior=None, mu_init=None, rho_init=None,
                 local_trick=False, regression=False, **kwargs):
        super().__init__(classes, regression)

        # if mu_init is None:
        #     mu_init = (-0.6, 0.6)

        # if rho_init is None:
        #     rho_init = -3

        if topology is None:
            topology = [400, 400]

        if prior is None:
            prior = Gaussian(0, 10)

        self.calculate_mmd = True
        self._prior = prior
        self.features = get_bayesian_network(topology, sample, classes,
                                             mu_init, rho_init, prior, 'mmd', local_trick)

        # ##################### non abbandonarmi più jary!!!!
        # ##################### Scusa :'(

    def _forward(self, x):

        mmd = 0
        for j, i in enumerate(self.features):
            if not isinstance(i, (BayesianLinearLayer, BayesianCNNLayer)):
                x = i(x)
            else:
                x, m = i(x, self.calculate_mmd)
                mmd += m

        return x, mmd

    def forward(self, x, samples=1):
        o = []
        mmds = []

        for i in range(samples):
            op, mmd = self._forward(x)
            o.append(op)
            mmds.append(mmd)

        o = torch.stack(o)

        mmds = torch.stack(mmds).mean()

        return o, mmds

    def layers(self):
        return chain(self.features, self.classificator)

    def eval_forward(self, x, samples=1):
        o, _ = self(x, samples=samples)
        return o


class Trainer(Wrapper):
    def __init__(self, model: nn.Module, train_data, test_data, optimizer):
        super().__init__(model, train_data, test_data, optimizer)

    def train_epoch(self, samples=1, **kwargs):
        losses = []

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=True, leave=False)
        progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

        train_true = []
        train_pred = []

        M = len(self.train_data)
        a = np.asarray([2 ** (M - i - 1) for i in range(M + 1)])
        b = 2 ** M - 1

        pi = a / b
        self.model.calculate_mmd = True

        for batch, (x, y) in progress_bar:

            train_true.extend(y.tolist())

            y = y.to(self.device)
            x = x.to(self.device)
            self.optimizer.zero_grad()

            out, mmd = self.model(x, samples=samples)
            out = out.mean(0)
            mmd /= x.shape[0]
            mmd *= pi[batch]

            if mmd == 0:
                self.model.calculate_mmd = False

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

            train_pred.extend(out.tolist())

            # ce = F.cross_entropy(out, y.to(self.device), reduction='mean')
            loss += mmd
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            # progress_bar.set_postfix(ce_loss=ce.item(), mmd_loss=mmd.item())

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
    #
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
    #     self.model.eval()
    #     for eps in self.epsilons:
    #
    #         H = []
    #         pred_label = []
    #         true_label = []
    #
    #         # self.model.eval()
    #         for i, (x, y) in enumerate(self.test_data):
    #             true_label.extend(y.tolist())
    #
    #             x = x.to(self.device)
    #             y = y.to(self.device)
    #
    #             self.model.zero_grad()
    #             x.requires_grad = True
    #             out, mmd = self.model(x, samples=samples)
    #             out = out.mean(0)
    #
    #             pred = (out.mean(0).argmax(dim=-1) == y).tolist()
    #
    #             ce = F.cross_entropy(out, y, reduction='mean')
    #             # loss = ce + mmd
    #             ce.backward()
    #
    #             perturbed_data = fgsm_attack(x, eps)
    #             out = self.model.eval_forward(perturbed_data, samples=samples)
    #             # out = out.mean(0)
    #             pred_label.extend(out.mean(0).argmax(dim=-1).tolist())
    #
    #             # a = compute_entropy(F.softmax(out, -1), True).tolist()
    #
    #             # print(a)
    #             # print(pred)
    #             # print([a[i] for i in range(len(a)) if pred[i] == 0 ], np.mean(a))
    #             # input()
    #             entropy = compute_entropy(F.softmax(out, -1)).mean(0)
    #
    #             # H.extend(compute_entropy(entropy).tolist())
    #             # H.extend(compute_entropy(F.softmax(out, -1)).tolist())
    #
    #         scores.append(metrics.f1_score(true_label, pred_label, average='micro'))
    #         HS.append(np.mean(H))
    #
    #     return HS, scores
