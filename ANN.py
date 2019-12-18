from itertools import chain

import torch
from torch import nn
from tqdm import tqdm

from base import Network, Wrapper, get_network, log_gaussian_loss, \
    cross_entropy_loss


class ANN(Network):
    def __init__(self, sample, classes, topology=None, regression=False, **kwargs):
        super().__init__(classes=classes, regression=regression)

        if topology is None:
            topology = [400, 400]

        self.features = get_network(topology, sample, classes)

        # self.regression = regression
        # self.classes = classes

    def forward(self, x, **kwargs):
        for j, i in enumerate(self.features):
            x = i(x)

        return x

    def layers(self):
        return chain(self.features, self.classificator)

    def eval_forward(self, x, **kwargs):
        return self.forward(x)


class Trainer(Wrapper):
    def __init__(self, model: ANN, train_data, test_data, optimizer):
        super().__init__(model, train_data, test_data, optimizer)

        self.regression = model.regression
        if model.regression:
            self.loss = log_gaussian_loss(model.classes)
        else:
            self.loss = cross_entropy_loss('mean')

    def train_epoch(self, **kwargs):
        losses = []

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=True)
        # progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

        train_true = []
        train_pred = []

        for batch, (x, y) in progress_bar:
            train_true.extend(y.tolist())
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            out = self.model(x)

            if self.regression:
                if self.model.classes == 1:
                    noise = self.model.noise.exp()
                    x = out
                    loss = self.loss_function(x, y, noise)
                else:
                    loss = self.loss_function(out[:, :1], y, out[:, 1:].exp())/x.shape[0]
            else:
                loss = self.loss_function(out, y)
                out = out.argmax(dim=-1)

            train_pred.extend(out.tolist())

            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix(ce_loss=loss.item())

        return losses, (train_true, train_pred)

    def test_evaluation(self, **kwargs):

        pred = []
        x_all = []
        y_all = []
        noises = []

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_data):
                y_all.extend(y.tolist())
                x_all.extend(x.tolist())

                out = self.model(x.to(self.device))

                if not self.regression:
                    out = out.argmax(dim=-1)
                    pred.extend(out.tolist())
                else:
                    pred.extend(out[:, 0].tolist())
                    if self.model.classes == 2:
                        noises.extend(out[:, 1].exp().tolist())

        if not self.regression:
            return y_all, pred

        if len(noises) == 0:
            noises = self.model.noise.exp().item()

        return x_all, y_all, pred, noises

    def snr_test(self, percentiles: list):
        return None

    # def rotation_test(self, **kwargs):
    #     ts_copy = copy(self.test_data.dataset.transform)
    #
    #     HS = []
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
    #         self.model.eval()
    #         with torch.no_grad():
    #             for i, (x, y) in enumerate(self.test_data):
    #                 true_label.extend(y.tolist())
    #
    #                 out = self.model.eval_forward(x.to(self.device))
    #                 pred_label.extend(out.argmax(dim=-1).tolist())
    #
    #                 H.extend(compute_entropy(F.softmax(out, -1)).tolist())
    #
    #         HS.append(np.mean(H))
    #
    #         scores.append(metrics.f1_score(true_label, pred_label, average='micro'))
    #
    #     self.test_data.dataset.transform = ts_copy
    #     return HS, scores
    #
    # def attack_test(self, **kwargs):
    #     HS = []
    #     scores = []
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
    #             true_label.extend(y.tolist())
    #
    #             x = x.to(self.device)
    #
    #             self.model.zero_grad()
    #             x.requires_grad = True
    #             out = self.model(x)
    #             pred = (out.argmax(dim=-1) == y).tolist()
    #
    #             loss = F.cross_entropy(out, y.to(self.device), reduction='mean')
    #             loss.backward()
    #
    #             perturbed_data = fgsm_attack(x, eps)
    #             out = self.model(perturbed_data)
    #
    #             pred_label.extend(out.argmax(dim=-1).tolist())
    #             a = compute_entropy(F.softmax(out, -1), True).tolist()
    #
    #             H.extend(compute_entropy(F.softmax(out, -1)).tolist())
    #             H.extend([a[i] for i in range(len(a)) if pred[i] == 0])
    #
    #         scores.append(metrics.f1_score(true_label, pred_label, average='micro'))
    #         HS.append(np.mean(H))
    #
    #     return HS, scores
    #
