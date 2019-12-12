from itertools import chain

import torch
from torch import nn
from tqdm import tqdm

from base import Network, Wrapper, Flatten, get_network
import torch.nn.functional as F

from bayesian_utils import b_drop


class Dropnet(Network):
    def __init__(self, sample, classes, topology=None, **kwargs):
        super().__init__()

        if topology is None:
            topology = [400, 400]

        # self.features = torch.nn.ModuleList()
        self.features = get_network(topology, sample, classes)

        # prev = input_size
        #
        # self.ann_type = ann_type
        # for i in topology:
        #     l = None
        #     if ann_type == 'linear':
        #         l = torch.nn.Linear(prev, i)
        #     else:
        #         l = torch.nn.Conv2d(in_channels=prev, out_channels=i, kernel_size=3)
        #
        #     self.features.append(l)
        #     prev = i
        #
        # input_image = input_image.unsqueeze(0)
        # if ann_type == 'cnn':
        #     for f in self.features:
        #         input_image = f(input_image)
        #     input_image = torch.flatten(input_image)
        #     prev = input_image.shape[0]
        # # print(prev)
        # self.classificator = nn.ModuleList([torch.nn.Linear(prev, classes)])

        # prev = input_size
        # input_image = input_image.unsqueeze(0)
        #
        # for j, i in enumerate(topology):
        #
        #     if isinstance(i, (tuple, list)) and i[0] == 'MP':
        #         l = torch.nn.MaxPool2d(i[1])
        #         input_image = l(input_image)
        #         prev = input_image.shape[1]
        #
        #     elif isinstance(i, (tuple, list)) and i[0] == 'AP':
        #         l = torch.nn.AvgPool2d(i[1])
        #         input_image = l(input_image)
        #         prev = input_image.shape[1]
        #
        #     elif isinstance(i, (tuple, list)):
        #         size, kernel_size = i
        #         l = torch.nn.Conv2d(in_channels=prev, out_channels=size, kernel_size=kernel_size)
        #
        #         input_image = l(input_image)
        #         prev = input_image.shape[1]
        #
        #     elif isinstance(i, int):
        #         if j > 0 and not isinstance(topology[j - 1], int):
        #             input_image = torch.flatten(input_image, 1)
        #             prev = input_image.shape[-1]
        #             self.features.append(Flatten())
        #
        #         size = i
        #         l = torch.nn.Linear(prev, i)
        #         prev = size
        #
        #     else:
        #         raise ValueError('Topology should be tuple for cnn layers, formatted as (num_kernels, kernel_size), '
        #                          'pooling layer, formatted as tuple ([\'MP\', \'AP\'], kernel_size, stride) '
        #                          'or integer, for linear layer. {} was given'.format(i))
        #
        #     self.features.append(l)
        #
        # if isinstance(topology[-1], (tuple, list)):
        #     input_image = torch.flatten(input_image, 1)
        #     prev = input_image.shape[-1]
        #     self.features.append(Flatten())
        #
        # self.features.append(torch.nn.Linear(prev, classes))

    def _forward(self, x):

        for j, i in enumerate(self.features):
            x = F.dropout(x, p=0.5, training=True, inplace=False)
            x = i(x)

            if j < len(self.features)-1:
                x = torch.relu(x)

        #     x = torch.relu(x)
        #
        # if self.ann_type == 'cnn':
        #     x = torch.flatten(x, 1)
        #
        # for i in self.classificator:
        #     x, prior, post = i(x)
        #     tot_post += post
        #     tot_prior += prior

        return x

    def forward(self, x, samples=1):
        o = []

        for i in range(samples):
            op = self._forward(x)
            o.append(op)

        o = torch.stack(o)

        return o

    def layers(self):
        return chain(self.features, self.classificator)

    def eval_forward(self, x, samples=1):
        o = self(x, samples=samples)
        return o


class Trainer(Wrapper):
    def __init__(self, model: nn.Module, train_data, test_data, optimizer):
        super().__init__(model, train_data, test_data, optimizer)

    def train_epoch(self, samples=1, **kwargs):
        losses = []

        weights = kwargs.get('weights', {})
        lreg = weights.get('lambda', 1e-5)

        self.model.train()
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=True)
        progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

        train_true = []
        train_pred = []

        for batch, (x_train, y_train) in progress_bar:

            train_true.extend(y_train.tolist())

            self.optimizer.zero_grad()

            out = self.model(x_train.to(self.device), samples=samples)
            out = out.mean(0)

            max_class = F.softmax(out, -1).argmax(dim=-1)
            train_pred.extend(max_class.tolist())

            loss = F.cross_entropy(out, y_train.to(self.device), reduction='mean')

            l2_reg = torch.tensor(0.).to(self.device)
            for param in self.model.parameters():
                l2_reg += torch.norm(param)

            loss += lreg*l2_reg

            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix(ce_loss=loss.item())

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

