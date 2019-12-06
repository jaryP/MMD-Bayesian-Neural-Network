from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

from base import BayesianLinearLayer, Network, Gaussian, Wrapper


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

        self.features = torch.nn.ModuleList()
        self._prior = prior

        prev = input_size

        for i in topology:
            self.features.append(
                BayesianLinearLayer(in_size=prev, out_size=i, mu_init=mu_init, divergence='mmd',
                                    rho_init=rho_init, prior=self._prior, local_rep_trick=local_trick))
            prev = i

        self.classificator = nn.ModuleList(
            [BayesianLinearLayer(in_size=prev, out_size=classes, mu_init=mu_init, rho_init=rho_init,
                                 prior=self._prior, divergence='mmd', local_rep_trick=local_trick)])

    def _forward(self, x):

        mmd = 0
        for j, i in enumerate(self.features):
            x, m = i(x)
            mmd += m
            x = torch.relu(x)

        x, m = self.classificator[0](x)

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
        progress_bar = tqdm(enumerate(self.train_data), total=len(self.train_data), disable=True)
        # progress_bar.set_postfix(mmd_loss='not calculated', ce_loss='not calculated')

        train_true = []
        train_pred = []

        for batch, (x_train, y_train) in progress_bar:
            train_true.extend(y_train.tolist())

            self.optimizer.zero_grad()

            out, mmd = self.model(x_train.to(self.device), samples=samples)
            out = out.mean(0)

            max_class = F.log_softmax(out, -1).argmax(dim=-1)
            train_pred.extend(max_class.tolist())

            ce = F.nll_loss(F.log_softmax(out, -1), y_train.to(self.device))
            loss = ce + mmd
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

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
#     mmd_w = weights.get('mmd', 1)
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
#         out, mmd = model.sample_forward(x_train.to(device), samples=train_samples)
#         out = out.mean(0)
#         mmd = (mmd * mmd_w)
#
#         max_class = F.log_softmax(out, -1).argmax(dim=-1)
#         train_pred.extend(max_class.tolist())
#
#         ce = F.nll_loss(F.log_softmax(out, -1), y_train.to(device))
#         loss = ce + mmd
#         losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
#
#         progress_bar.set_postfix(mmd_loss=mmd.item(), ce_loss=ce.item())
#
#     test_pred = []
#     test_true = []
#
#     model.eval()
#     with torch.no_grad():
#         for i, (x_test, y_test) in enumerate(test_dataset):
#             test_true.extend(y_test.tolist())
#
#             out, mmd = model.sample_forward(x_test.to(device), samples=test_samples)
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

    ann = BMMD(784, 10, local_trick=True)
    ann.cuda()
    trainer = Trainer(ann, train_loader, test_loader, torch.optim.Adam(ann.parameters(), 1e-3))

    for i in range(10):
       a, _, (test_true, test_pred) = trainer.train_step(test_samples=2, train_samples=2)
       f1 = metrics.f1_score(test_true, test_pred, average='micro')

       print(f1)