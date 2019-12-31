from abc import ABC, abstractmethod
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn import metrics
from torch.distributions import Normal
from tqdm import tqdm

from bayesian_utils import BayesianCNNLayer, BayesianLinearLayer


#
# def pairwise_distances(x, y):
#     x_norm = (x ** 2).sum(1).view(-1, 1)
#     y_norm = (y ** 2).sum(1).view(1, -1)
#
#     dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
#     return torch.clamp(dist, 0.0, np.inf)
# #
# #
# def compute_kernel(x, y):
#     dim = x.size(1)
#     # d = torch.exp(- torch.mul(torch.cdist(x, y).mean(1), 1/float(dim))).mean()
#     d = torch.exp(- torch.mul(pairwise_distances(x, y).mean(1), 1 / float(dim))).mean()
#     return d
# #
# #
# def compute_mmd(x, y):
#     x_kernel = compute_kernel(x, x)
#     y_kernel = compute_kernel(y, y)
#     xy_kernel = compute_kernel(x, y)
#     return x_kernel + y_kernel - 2 * xy_kernel


class AngleRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return T.functional.rotate(x, self.angle)


class AddNoise:
    def __init__(self, noise):
        self.noise = noise

    def __call__(self, x):
        return x + torch.randn(x.size()) * self.noise


# FGSM attack code
def fgsm_attack(image, epsilon):
    if epsilon == 0:
        return image
    # Collect the element-wise sign of the data gradient
    sign_data_grad = image.grad.data.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    # print(epsilon, perturbed_image.mean(), image.mean())
    return perturbed_image


def log_gaussian_loss(out_dim):
    # def loss_function(x, y, sigma):
    #     log_coeff = -1 * torch.log(sigma + 1e-12) - 0.5 * ((x - y) ** 2 / sigma ** 2)
    #     return (- log_coeff).sum()

    def loss_function(x, y, sigma):
        exponent = -0.5 * (x - y) ** 2 / sigma ** 2
        log_coeff = -torch.log(sigma+1e-12) - 0.5 * np.log(2 * np.pi)
        return -(log_coeff + exponent).sum()

    return loss_function


def cross_entropy_loss(reduction):
    def loss_function(x, y):
        return F.cross_entropy(x, y, reduction=reduction)
    return loss_function


def epistemic_aleatoric_uncertainty(x):
    t = x.shape[0]
    classes = x.shape[-1]
    # x = np.e ** x / sum(np.e ** x, -1)
    # x = x.mean(0)
    # x[[0, 1, 2]] = x[]
    x = np.moveaxis(x, 0, 1)

    # x = torch.nn.functional.softmax(x, -1)
    # print(x.shape, x[0])
    # p_hat = x.detach().cpu().numpy()
    # aleatorics = []
    unc = []

    for p in x:
        aleatoric = np.zeros((classes, classes))
        epistemic = np.zeros((classes, classes))

        np.argmax(p)
        # print(p.shape)
        # print(p)
        p = np.e ** p / sum(np.e ** p, 1)
        # print(p)
        # print(p.sum(-1))
        # input()
        p_mean = p.mean(0)

        for p_hat in p:
            # print(np.e ** p_hat / sum(np.e ** p_hat, -1), np.sum(np.e ** p_hat / sum(np.e ** p_hat, -1)))
            # print(p_hat, p_hat.sum())
            a = np.diag(p_hat) - np.outer(p_hat, p_hat)/p_hat.shape[0]

            # print(np.argmax(p))
            # print(a[np.argmax(p), np.argmax(p)])
            aleatoric += a

            p_hat -= p_mean
            epistemic += np.outer(p_hat, p_hat)

        unc.append((np.mean(aleatoric/t), np.mean(epistemic/t)))
        # print(np.mean(aleatoric/t), np.mean(epistemic/t))
        # input()
        # p = p_hat[i]
        # x = x.mean(0)

        # p_hat = x
        # # p_hat = np.max(x, -1)
        # # print(p_hat)
        # epistemic = np.mean((p_hat) ** 2, axis=0) - np.mean((p_hat), axis=0) ** 2
        # # epistemic = np.mean((1-p_hat) ** 2, axis=0) - np.mean((1-p_hat), axis=0) ** 2
        # # epistemic += epistemic
        # print (epistemic)
        # aleatoric = np.mean((p_hat) * (1-(p_hat)), axis = 0)
        # # aleatoric = np.mean((1-p_hat), axis = 0) ** 2
        # # aleatoric += aleatoric
        #
        # print(epistemic, aleatoric)
        # input()
    # res = np.mean(x, axis=0)
    # aleatoric = np.diag(res) - res.T.dot(res) / res.shape[0]
    # input(aleatoric.sum())

    # p = p
    # aleatoric = 0
    # for i in range(len(x)):
    # print()
    # print(np.diag(p_hat).shape)
    # aleatoric = np.diag(p_hat) - p_hat.T.dot(p_hat)/p_hat.shape[0]
    # aleatoric /= t
    # input(aleatoric)
    #
    # epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    # aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    #
    # print(epistemic, aleatoric)
    # # aleatorics.append(ale)
    # ## input(aleatoric)
    # # input(aleatoric)
    #
    # p_bar = np.sum(p_hat, keepdims=True, axis=-1)/t
    # diff = p_hat - p_bar
    #
    # epistemic = diff.T.dot(diff)/p_bar.shape[0]
    #
    # print(np.sum(aleatoric, keepdims=True), np.sum(epistemic, keepdims=True))
    # input()
    return np.asarray(unc)


def compute_entropy(preds, sum=True):
    l = torch.log10(preds + 1e-12) * preds
    if sum:
        return -torch.sum(l, 1)
    else:
        return -l
    # return -(torch.log2(preds + 1e-10) * preds).sum(dim=1)


def get_bayesian_network(topology, input_image, classes, mu_init, rho_init, prior, divergence, local_trick,
                         bias=True, **kwargs):

    features = torch.nn.ModuleList()
    # self._prior = prior
    # print(mu_init)
    prev = input_image.shape[0]
    input_image = input_image.unsqueeze(0)
    ll_conv = False

    for j, i in enumerate(topology):

        if isinstance(i, (tuple, list)) and i[0] == 'MP':
            l = torch.nn.MaxPool2d(i[1])
            input_image = l(input_image)
            prev = input_image.shape[1]
            ll_conv = True

        elif isinstance(i, str) and i.lower() == 'relu':
            l = torch.nn.ReLU()

        elif isinstance(i, float):
            l = torch.nn.Dropout(p=0.5)

        elif isinstance(i, (tuple, list)) and i[0] == 'AP':
            l = torch.nn.AvgPool2d(i[1])
            input_image = l(input_image)
            prev = input_image.shape[1]
            ll_conv = True

        elif isinstance(i, (tuple, list)):
            size, kernel_size = i
            l = BayesianCNNLayer(in_channels=prev, kernels=size, kernel_size=kernel_size,
                                 mu_init=mu_init, divergence=divergence, local_rep_trick=local_trick,
                                 rho_init=rho_init, prior=prior, **kwargs)

            input_image = l(input_image)[0]
            prev = input_image.shape[1]

        elif isinstance(i, int):
            if ll_conv:
                input_image = torch.flatten(input_image, 1)
                prev = input_image.shape[-1]
                features.append(Flatten())
            ll_conv = False

            size = i
            l = BayesianLinearLayer(in_size=prev, out_size=size, mu_init=mu_init, divergence=divergence,
                                    rho_init=rho_init, prior=prior, local_rep_trick=local_trick, use_bias=bias,
                                    **kwargs)
            prev = size

        else:
            raise ValueError('Topology should be tuple for cnn layers, formatted as (num_kernels, kernel_size), '
                             'pooling layer, formatted as tuple ([\'MP\', \'AP\'], kernel_size, stride) '
                             'or integer, for linear layer. {} was given'.format(i))

        features.append(l)

    if isinstance(topology[-1], (tuple, list)):
        input_image = torch.flatten(input_image, 1)
        prev = input_image.shape[-1]
        features.append(Flatten())

    features.append(BayesianLinearLayer(in_size=prev, out_size=classes, mu_init=mu_init, rho_init=rho_init,
                                        prior=prior, divergence=divergence, local_rep_trick=local_trick, **kwargs))
    return features


def get_network(topology, input_image, classes, bias=True):
    features = torch.nn.ModuleList()

    prev = input_image.shape[0]
    input_image = input_image.unsqueeze(0)
    ll_conv = False

    for j, i in enumerate(topology):

        if isinstance(i, (tuple, list)) and i[0] == 'MP':
            l = torch.nn.MaxPool2d(i[1])
            input_image = l(input_image)
            prev = input_image.shape[1]
            ll_conv = True

        elif isinstance(i, str) and i.lower() == 'relu':
            l = torch.nn.ReLU()

        elif isinstance(i, float):
            l = torch.nn.Dropout(p=0.5)

        elif isinstance(i, (tuple, list)) and i[0] == 'AP':
            l = torch.nn.AvgPool2d(i[1])
            input_image = l(input_image)
            prev = input_image.shape[1]
            ll_conv = True

        elif isinstance(i, (tuple, list)):
            size, kernel_size = i
            l = torch.nn.Conv2d(in_channels=prev, out_channels=size, kernel_size=kernel_size, bias=bias)

            input_image = l(input_image)
            prev = input_image.shape[1]
            ll_conv = True

        elif isinstance(i, int):
            if ll_conv:
                input_image = torch.flatten(input_image, 1)
                prev = input_image.shape[-1]
                features.append(Flatten())

            ll_conv = False
            size = i
            l = torch.nn.Linear(prev, i)
            prev = size
        else:
            # 'Supported type for topology: \n' \
            # 'List: []'
            raise ValueError('Topology should be tuple for cnn layers, formatted as (num_kernels, kernel_size), '
                             'pooling layer, formatted as tuple ([\'MP\', \'AP\'], kernel_size, stride) '
                             'or integer, for linear layer. {} was given'.format(i))

        features.append(l)

    if ll_conv:
        input_image = torch.flatten(input_image, 1)
        prev = input_image.shape[-1]
        features.append(Flatten())

    features.append(torch.nn.Linear(prev, classes))

    return features


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


# PRIORS


class Gaussian(object):
    def __init__(self, mu=0, sigma=5):
        self.mu = mu
        self.sigma = sigma
        self.inner_gaussian = Normal(mu, sigma)
        # self.gaussian = torch.distributions.Normal(mu, sigma)

    def sample(self, size):
        return self.inner_gaussian.rsample(size)
        # return self.mu + self.sigma * Normal(0, 1).sample(size)

    def log_prob(self, x):
        return self.inner_gaussian.log_prob(x)


class Laplace(object):
    def __init__(self, mu=0, scale=1):
        self.mu = mu
        self.scale = scale
        self.distribution = torch.distributions.laplace.Laplace(mu, scale)

    def sample(self, size):
        return self.distribution.rsample(size)
        # return self.mu + self.sigma * Normal(0, 1).sample(size)

    def log_prob(self, x):
        return self.distribution.log_prob(x)


class ScaledMixtureGaussian(object):
    def __init__(self, pi, s1, s2, mu1=0, mu2=0):
        self.pi = pi
        self.s1 = s1
        self.s2 = s2
        self.mu1 = mu1
        self.mu2 = mu2
        self.gaussian1 = Gaussian(mu1, s1)
        self.gaussian2 = Gaussian(mu2, s2)

    def sample(self, size):
        return self.pi * self.gaussian1.sample(size) + (1 - self.pi) * self.gaussian2.sample(size)

    def log_prob(self, x):
        return self.pi * self.gaussian1.log_prob(x) + (1 - self.pi) * self.gaussian2.log_prob(x)


# Utils

class Network(nn.Module, ABC):
    def __init__(self, classes, regression=False):
        super().__init__()
        self.classes = classes
        self.regression = regression

        if regression:
            # if classes == 1:
            self.noise = nn.Parameter(torch.tensor(0.0))

    @abstractmethod
    def layers(self):
        pass

    @abstractmethod
    def eval_forward(self, x, **kwargs):
        pass


class Wrapper(ABC):
    epsilons = [0, 0.01] #, 0.05, .1, .2, .5, .8]
    rotations = [0, 15] #, 30, 45, 90, 180, 270]

    def __init__(self, model: nn.Module, train_data, test_data, optimizer):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.device = next(model.parameters()).device

        self.regression = model.regression

        if model.regression:
            self.loss_function = log_gaussian_loss(model.classes)
        else:
            self.loss_function = cross_entropy_loss('mean')

    def train_step(self, **kwargs):
        losses, train_res = self.train_epoch(**kwargs)
        test_res = self.test_evaluation(**kwargs)
        return losses, train_res, test_res

    def ce_loss(self, x, y):
        pass

    @abstractmethod
    def train_epoch(self, **kwargs):
        pass

    @abstractmethod
    def test_evaluation(self, **kwargs):
        pass

    @abstractmethod
    def snr_test(self, percentiles: list):
        pass

    # @abstractmethod
    # def attack_test(self, **kwargs):
    #     pass

    def rotation_test(self, samples=1):

        ts_copy = copy(self.test_data.dataset.transform)

        HS = []
        DIFF = []
        scores = []
        self.model.eval()

        for angle in tqdm(self.rotations, desc='Rotation test'):
            ts = T.Compose([AngleRotation(angle), ts_copy])
            self.test_data.dataset.transform = ts

            H = []
            pred_label = []
            true_label = []
            diff = []

            self.model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(self.test_data):
                    true_label.extend(y.tolist())

                    out = self.model.eval_forward(x.to(self.device), samples=samples)

                    out_m = out.mean(0)
                    pred_label.extend(out_m.argmax(dim=-1).tolist())

                    top_score, top_label = torch.topk(F.softmax(out.mean(0), -1), 2)

                    H.extend(top_score[:, 0].tolist())
                    diff.extend(((top_score[:, 0] - top_score[:, 1]) ** 2).tolist())

            p_hat = np.asarray(H)

            mean_diff = np.mean(diff)

            epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
            aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

            entropy = aleatoric + epistemic

            HS.append((entropy, np.mean(H)))
            DIFF.append(mean_diff)
            scores.append(metrics.f1_score(true_label, pred_label, average='micro'))

        self.test_data.dataset.transform = ts_copy
        return HS, DIFF, scores

    def attack_test(self, samples=1):

        HS = []
        DIFF = []
        scores = []

        self.model.eval()
        for eps in tqdm(self.epsilons, desc='Attack test'):

            H = []
            pred_label = []
            true_label = []
            diff = []

            self.model.eval()
            for i, (x, y) in enumerate(self.test_data):
                true_label.extend(y.tolist())

                x = x.to(self.device)
                y = y.to(self.device)

                self.model.zero_grad()
                x.requires_grad = True
                out = self.model.eval_forward(x.to(self.device), samples=samples)

                ce = F.cross_entropy(out.mean(0), y, reduction='mean')
                ce.backward()

                perturbed_data = fgsm_attack(x, eps)
                out = self.model.eval_forward(perturbed_data, samples=samples)

                out_m = out.mean(0)
                pred_label.extend(out_m.argmax(dim=-1).tolist())

                top_score, top_label = torch.topk(F.softmax(out.mean(0), -1), 2)

                H.extend(top_score[:, 0].tolist())
                diff.extend(((top_score[:, 0] - top_score[:, 1]) ** 2).tolist())

            p_hat = np.asarray(H)

            mean_diff = np.mean(diff)

            epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
            aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)

            entropy = aleatoric + epistemic

            HS.append((entropy, np.mean(H)))
            DIFF.append(mean_diff)
            scores.append(metrics.f1_score(true_label, pred_label, average='micro'))

        return HS, DIFF, scores
