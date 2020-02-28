from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from base import Wrapper, get_network, Network


class Dropnet(Network):
    def __init__(self, sample, classes, topology=None, regression=False, **kwargs):
        super().__init__(classes, regression)

        self.p = kwargs.get('drop', 0.5)

        if topology is None:
            topology = [400, 400]

        self.features = get_network(topology, sample, classes)

    def _forward(self, x):

        for j, i in enumerate(self.features):
            if isinstance(i, (torch.nn.Linear, torch.nn.Conv2d)):
                x = F.dropout(x, p=self.p, training=True, inplace=False)
            x = i(x)

        return x

    def forward(self, x, samples=1):
        o = []

        for i in range(samples):
            op = self._forward(x)
            o.append(op)

        o = torch.stack(o)

        return o

    def eval_forward(self, x, samples=1):
        o = self(x, samples=samples)
        return o


class Trainer(Wrapper):
    def __init__(self, model: nn.Module, train_data, test_data, optimizer, wd=1e-5, **kwargs):
        super().__init__(model, train_data, test_data, optimizer)
        self.wd = wd

    def train_epoch(self, samples=1, **kwargs):
        losses = []

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=False, leave=False)

        train_true = []
        train_pred = []

        for batch, (x, y) in progress_bar:
            x = x.to(self.device)
            y = y.to(self.device)
            train_true.extend(y.tolist())

            self.optimizer.zero_grad()

            out = self.model(x, samples=samples)

            if self.regression:
                out = out.mean(0)

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

            train_pred.extend(out.tolist())

            if self.wd != 0:
                l2_reg = torch.tensor(0.).to(self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)

                loss += self.wd*l2_reg

            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix(ce_loss=loss.item())

        return losses, (train_true, train_pred)

    def train_step(self, train_samples=1, test_samples=1, **kwargs):
        losses, train_res = self.train_epoch(samples=train_samples)
        test_res = self.test_evaluation(samples=test_samples)
        return losses, train_res, test_res
