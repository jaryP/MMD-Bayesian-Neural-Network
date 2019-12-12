from itertools import chain

import torch
from torch import nn
from tqdm import tqdm

from base import Network, Wrapper, Flatten, get_network
import torch.nn.functional as F


class ANN(Network):
    def __init__(self, sample, classes, topology=None, **kwargs):
        super().__init__()

        if topology is None:
            topology = [400, 400]

        self.features = get_network(topology, sample, classes)

    def forward(self, x):

        for j, i in enumerate(self.features):
            x = i(x)

            if j < len(self.features)-1:
                x = torch.relu(x)

        return x

    def layers(self):
        return chain(self.features, self.classificator)

    def eval_forward(self, x, **kwargs):
        return self.forward(x)


class Trainer(Wrapper):
    def __init__(self, model: nn.Module, train_data, test_data, optimizer):
        super().__init__(model, train_data, test_data, optimizer)

    def train_epoch(self, **kwargs):
        losses = []

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=True)
        # progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

        train_true = []
        train_pred = []

        for batch, (x_train, y_train) in progress_bar:
            train_true.extend(y_train.tolist())

            self.optimizer.zero_grad()

            out = self.model(x_train.to(self.device))

            max_class = F.log_softmax(out, -1).argmax(dim=-1)
            train_pred.extend(max_class.tolist())

            loss = F.cross_entropy(out, y_train.to(self.device), reduction='mean')

            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix(ce_loss=loss.item())

        return losses, (train_true, train_pred)

    def test_evaluation(self, **kwargs):

        test_pred = []
        test_true = []

        self.model.eval()
        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(self.test_data):
                test_true.extend(y_test.tolist())

                out = self.model(x_test.to(self.device))
                out = out.argmax(dim=-1)
                test_pred.extend(out.tolist())

        return test_true, test_pred

    def snr_test(self, percentiles: list):
        return None

