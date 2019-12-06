from itertools import chain

import torch
from sklearn import metrics
from torch import nn
from torchvision.transforms import transforms
from torchvision import datasets
from tqdm import tqdm

from base import Network, Wrapper
import torch.nn.functional as F


class ANN(Network):
    def __init__(self, input_size, classes, topology=None, **kwargs):
        super().__init__()

        if topology is None:
            topology = [400, 400]

        self.features = torch.nn.ModuleList()

        prev = input_size

        for i in topology:
            self.features.append(
                torch.nn.Linear(prev, i))
            prev = i

        self.classificator = nn.ModuleList([torch.nn.Linear(prev, classes)])

    def forward(self, x, **kwargs):

        for j, i in enumerate(self.features):
            r = i(x)
            x = torch.relu(r)

        x = self.classificator[0](x)
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

            loss = F.nll_loss(F.log_softmax(out, -1), y_train.to(self.device))

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


# def sample_forward(self, x, samples=1, task=None):
#     o = []
#     mmds = []
#
#     for i in range(samples):
#         op, mmd = self(x, task=task)
#         o.append(op)
#         mmds.append(mmd)
#
#     o = torch.stack(o)
#
#     mmds = torch.stack(mmds).mean()
#
#     return o, mmds

if __name__ == '__main__':
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

    ann = ANN(784, 10)
    ann.cuda()
    trainer = Trainer(ann, train_loader, test_loader, torch.optim.Adam(ann.parameters(), 1e-3))

    for i in range(10):
       a, _, (test_true, test_pred) = trainer.train_step(cacca=20)
       f1 = metrics.f1_score(test_true, test_pred, average='micro')

       print(f1)