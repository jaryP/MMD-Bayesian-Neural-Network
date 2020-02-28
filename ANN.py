import torch
from tqdm import tqdm

from base import Wrapper, get_network, log_gaussian_loss, \
    cross_entropy_loss, Network


class ANN(Network):
    def __init__(self, sample, classes, topology=None, regression=False, **kwargs):
        super().__init__(classes=classes, regression=regression)

        if topology is None:
            topology = [400, 400]

        self.features = get_network(topology, sample, classes)

    def forward(self, x, **kwargs):
        for j, i in enumerate(self.features):
            x = i(x)

        return x

    def eval_forward(self, x, **kwargs):
        return self.forward(x)


class Trainer(Wrapper):
    def __init__(self, model: ANN, train_data, test_data, optimizer, **kwargs):
        super().__init__(model, train_data, test_data, optimizer)

        self.regression = model.regression
        if model.regression:
            self.loss = log_gaussian_loss(model.classes)
        else:
            self.loss = cross_entropy_loss('mean')

    def train_epoch(self, **kwargs):
        losses = []

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=False, leave=False)

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
