from abc import ABC, abstractmethod
from copy import copy

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn import metrics
from sklearn.calibration import calibration_curve
from torch.distributions import Normal
from tqdm import tqdm

from bayesian_utils import BayesianCNNLayer, BayesianLinearLayer


class percentageRotation:
    def __init__(self, percentage):
        self.percentage = percentage

    def __call__(self, x):
        return T.functional.rotate(x, self.percentage)


class PixelShuffle:
    def __init__(self, percentage):
        if percentage < 0 or percentage > 1:
            raise ValueError('percentage should be between 0 and 1, {} wasa given'.format(percentage))

        self.percentage = percentage
        self.pixels_map = None

    def shuffle_pixels(self, x):
        x1 = x.copy()
        if self.pixels_map is None:
            w, h = x.size
            pxs = []
            for x in range(w):
                for y in range(h):
                    pxs.append((x, y))

            ln = len(pxs)
            idx = np.arange(ln)

            pixels_map = \
                zip(np.random.choice(idx, int(ln*self.percentage)), np.random.choice(idx, int(ln*self.percentage)))

            self.pixels_map = [(pxs[i], x1.getpixel(pxs[j])) for i, j in pixels_map]

        for a, b in self.pixels_map:
            x1.putpixel(a, b)

        return x1

    def __call__(self, x):
        x1 = self.shuffle_pixels(x)

        # if self.percentage > 0:
        #     import matplotlib.pyplot as plt
        #     plt.subplot(1, 1, 1)
        #     # plt.imshow(x, interpolation='none')
        #     # plt.subplot(2, 1, 2)
        #     plt.imshow((np.asarray(x1)-np.asarray(x))/np.max(np.asarray(x1)-np.asarray(x)), interpolation='none')
        #     plt.show()
        return x1


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
    return perturbed_image


def log_gaussian_loss(out_dim):
    def loss_function(x, y, sigma):
        exponent = -0.5 * (x - y) ** 2 / sigma ** 2
        log_coeff = -torch.log(sigma+1e-12) - 0.5 * np.log(2 * np.pi)
        return -(log_coeff + exponent).sum()

    return loss_function


def cross_entropy_loss(reduction):
    def loss_function(x, y):
        _x = F.log_softmax(x, -1)
        if _x.dim() == 3:
            _x = _x.mean(0)
        return F.nll_loss(_x, y, reduction=reduction)
        # return F.cross_entropy(x, y, reduction=reduction)
    return loss_function


def epistemic_aleatoric_uncertainty(x):

    if x.dim() == 2:
        x = x.unsqueeze(0)

    p = torch.softmax(x, -1)
    p_hat = torch.mean(p, 0)

    p = p.detach().cpu().numpy()
    p_hat = p_hat.detach().cpu().numpy()

    t = p.shape[0]
    classes = p.shape[-1]
    # x = np.e ** x / sum(np.e ** x, -1)
    # x = x.mean(0)
    # x[[0, 1, 2]] = x[]
    # x = np.moveaxis(x, 0, 1)

    # x = torch.nn.functional.softmax(x, -1)
    # print(x.shape, x[0])
    # p_hat = x.detach().cpu().numpy()
    # aleatorics = []

    # unc = []
    # variance = np.zeros((classes, classes))
    vars = []

    for _p, _p_hat in zip(p, p_hat):
        variance = np.zeros((classes, classes))

        for ip, ip_hat in zip(_p, _p_hat):
            aleatoric = np.diag(ip) - np.outer(ip, ip)
            d = ip - ip_hat
            epistemic = np.outer(d, d)

            variance += aleatoric + epistemic

        variance = variance/t
        det = np.linalg.det(variance)
        var = classes/2 * np.log(2*np.e*np.pi) + 0.5 * np.log(det)

        vars.append(var)

        continue

        # np.argmax(p)
        # # print(p.shape)
        # # print(p)
        # p = np.e ** p / sum(np.e ** p, 1)
        # # print(p)
        # # print(p.sum(-1))
        # # input()
        # p_mean = p.mean(0)
        #
        # for p_hat in p:
        #     # print(np.e ** p_hat / sum(np.e ** p_hat, -1), np.sum(np.e ** p_hat / sum(np.e ** p_hat, -1)))
        #     # print(p_hat, p_hat.sum())
        #     a = np.diag(p_hat) - np.outer(p_hat, p_hat)/p_hat.shape[0]
        #
        #     # print(np.argmax(p))
        #     # print(a[np.argmax(p), np.argmax(p)])
        #     aleatoric += a
        #
        #     p_hat -= p_mean
        #     epistemic += np.outer(p_hat, p_hat)
        #
        # unc.append((np.mean(aleatoric/t), np.mean(epistemic/t)))
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
    return vars


def compute_entropy(preds, sum=True):
    l = torch.log10(preds + 1e-12) * preds
    if sum:
        return -torch.sum(l, 1)
    else:
        return -l
    # return -(torch.log2(preds + 1e-10) * preds).sum(dim=1)


def get_bayesian_network(topology, input_image, classes, mu_init, rho_init, prior, divergence, local_trick,
                         posterior_type, bias=True, **kwargs):

    features = torch.nn.ModuleList()
    # self._prior = prior
    # print(mu_init)
    prev = input_image.shape[0]
    input_image = input_image.unsqueeze(0)
    ll_conv = False

    for j, i in enumerate(topology):

        if isinstance(i, (tuple, list)) and i[0] == 'MP':
            # size, kernel_size, stride = i

            l = torch.nn.MaxPool2d(kernel_size=i[1], stride=i[2])
            input_image = l(input_image)
            prev = input_image.shape[1]
            ll_conv = True

        elif isinstance(i, str) and i.lower() == 'relu':
            l = torch.nn.ReLU()

        elif isinstance(i, float):
            l = torch.nn.Dropout(p=0.5)

        elif isinstance(i, (tuple, list)) and i[0] == 'AP':
            l = torch.nn.AvgPool2d(kernel_size=i[1], stride=i[2])
            input_image = l(input_image)
            prev = input_image.shape[1]
            ll_conv = True

        elif isinstance(i, (tuple, list)):
            size, kernel_size, stride, padding = i

            l = BayesianCNNLayer(in_channels=prev, kernels=size, kernel_size=kernel_size, posterior_type=posterior_type,
                                 mu_init=mu_init, divergence=divergence, local_rep_trick=local_trick, stride=stride,
                                 rho_init=rho_init, prior=prior, padding=padding, **kwargs)

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
                                    posterior_type=posterior_type, **kwargs)
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
                                        prior=prior, divergence=divergence, local_rep_trick=local_trick,
                                        posterior_type=posterior_type, **kwargs))
    return features


def get_network(topology, input_image, classes, bias=True):
    features = torch.nn.ModuleList()

    prev = input_image.shape[0]
    input_image = input_image.unsqueeze(0)
    ll_conv = False

    for j, i in enumerate(topology):

        if isinstance(i, (tuple, list)) and i[0] == 'MP':
            l = torch.nn.MaxPool2d(kernel_size=i[1], stride=i[2])
            input_image = l(input_image)
            prev = input_image.shape[1]
            ll_conv = True

        elif isinstance(i, str) and i.lower() == 'relu':
            l = torch.nn.ReLU()

        elif isinstance(i, float):
            l = torch.nn.Dropout(p=0.5)

        elif isinstance(i, (tuple, list)) and i[0] == 'AP':
            l = torch.nn.AvgPool2d(kernel_size=i[1], stride=i[2])
            input_image = l(input_image)
            prev = input_image.shape[1]
            ll_conv = True

        elif isinstance(i, (tuple, list)):
            size, kernel_size, stride, padding = i
            l = torch.nn.Conv2d(in_channels=prev, out_channels=size, stride=stride,
                                kernel_size=kernel_size, bias=False, padding=padding)

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


class Uniform:
    def __init__(self, a, b):
        self.dist = torch.distributions.uniform.Uniform(a, b)

    def sample(self, size):
        return self.dist.rsample(size)

    def log_prob(self, x):
        return self.dist.log_prob(x)

# Utils


class Network(nn.Module, ABC):
    def __init__(self, classes, regression=False):
        super().__init__()
        self.classes = classes
        self.regression = regression
        self.features = []

        if regression:
            self.noise = nn.Parameter(torch.tensor(0.0))

    @abstractmethod
    def eval_forward(self, x, **kwargs):
        pass

    def set_mask(self, p):
        for i in self.features:
            if isinstance(i, (BayesianLinearLayer, BayesianCNNLayer)):
                i.set_mask(p)


class Wrapper(ABC):
    epsilons = [0, 0.01, 0.05, .1]  # , .2, .5]
    shuffle_percentage = [0, .1, .2, .5, .8]

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

    @abstractmethod
    def train_epoch(self, **kwargs):
        pass

    def test_evaluation(self, samples, **kwargs):

        test_pred = []
        test_true = []

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(self.test_data), leave=False, total=len(self.test_data)):
                test_true.extend(y.tolist())

                out = self.model.eval_forward(x.to(self.device), samples=samples)
                out = torch.softmax(out, -1)

                if out.dim() > 2:
                    out = out.mean(0)

                out = out.argmax(dim=-1)
                # out = out.sum(0)
                test_pred.extend(out.tolist())

        return test_true, test_pred

    def shuffle_test(self, samples=1):

        ts_copy = copy(self.test_data.dataset.transform)

        HS = []
        DIFF = []
        scores = []
        self.model.eval()

        for percentage in tqdm(self.shuffle_percentage, desc='Pixel Shuffle test'):
            ts = T.Compose([PixelShuffle(percentage), ts_copy])
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

                    out = torch.softmax(out, -1)

                    a = epistemic_aleatoric_uncertainty(out)
                    H.extend(a)

                    if out.dim() > 2:
                        out = out.mean(0)

                    pred_label.extend(out.argmax(dim=-1).tolist())

                    top_score, top_label = torch.topk(out, 2)

                    # H.extend(top_score[:, 0].tolist())
                    diff.extend(((top_score[:, 0] - top_score[:, 1]) ** 2).tolist())

            H = -np.log(np.mean(H))
            # print(percentage, np.log(np.mean(a)))
            # input()

            # p_hat = np.asarray(H)
            #
            # mean_diff = np.mean(diff)
            #
            # epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
            # aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
            #
            # entropy = aleatoric + epistemic

            HS.append(H)

            # DIFF.append(mean_diff)
            scores.append(metrics.f1_score(true_label, pred_label, average='micro'))

        self.test_data.dataset.transform = ts_copy
        return HS, DIFF, scores

    def attack_test(self, samples=1):

        HS = []
        DIFF = []
        scores = []

        self.model.eval()
        loss = cross_entropy_loss('mean')

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
                ce = loss(out, y)
                ce.backward()

                # out = torch.softmax(out, -1)
                # if out.dim() > 2:
                #     out = out.mean(0)
                # else:
                #     out_m = out

                # pred_label.extend(out.argmax(dim=-1).tolist())

                perturbed_data = fgsm_attack(x, eps)

                out = self.model.eval_forward(perturbed_data, samples=samples)
                out = torch.softmax(out, -1)

                a = epistemic_aleatoric_uncertainty(out)
                H.extend(a)

                if out.dim() > 2:
                    out = out.mean(0)
                # else:
                #     out_m = out

                pred_label.extend(out.argmax(dim=-1).tolist())

                top_score, top_label = torch.topk(out, 2)

                H.extend(top_score[:, 0].tolist())
                diff.extend(((top_score[:, 0] - top_score[:, 1]) ** 2).tolist())

            H = np.mean(H)

            # p_hat = np.asarray(H)

            mean_diff = np.mean(diff)

            # epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
            # aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
            #
            # entropy = aleatoric + epistemic

            HS.append(H)
            DIFF.append(mean_diff)
            scores.append(metrics.f1_score(true_label, pred_label, average='micro'))

        return HS, DIFF, scores

    def reliability_diagram(self, samples=1, bins=10):

        def get_diagram_data(y, p, n_bins):

            n_bins = float(n_bins)  # a float to take care of division

            # we'll append because some bins might be empty
            mean_predicted_values = np.empty((0,))
            true_fractions = np.zeros((0,))

            for b in range(1, int(n_bins) + 1):
                i = np.logical_and(p <= b / n_bins, p > (b - 1) / n_bins)  # indexes for p in the current bin

                # skip bin if empty
                if np.sum(i) == 0:
                    continue

                mean_predicted_value = np.mean(p[i])
                # print "***", np.sum( y[i] ), np.sum( i )
                true_fraction = np.sum(y[i]) / np.sum(i)  # y are 0/1; i are logical and evaluate to 0/1

                # print
                # mean_predicted_value, true_fraction

                mean_predicted_values = np.hstack((mean_predicted_values, mean_predicted_value))
                true_fractions = np.hstack((true_fractions, true_fraction))

            return (mean_predicted_values, true_fractions)

        y_prob = []
        y_true = []
        y_pred = []

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_data):
                y_true.extend(y.tolist())

                out = self.model.eval_forward(x.to(self.device), samples=samples)
                out = torch.softmax(out, -1)

                if out.dim() > 2:
                    out = out.mean(0)

                prob, pred = torch.topk(out, 1, -1)
                y_prob.extend(prob.tolist())
                y_pred.extend(pred.tolist())

        y_true = np.asarray(y_true)[:, None]
        y_prob = np.asarray(y_prob)
        y_pred = np.asarray(y_pred)

        prob_pred = np.empty((0,))
        prob_true = np.zeros((0,))
        ece = 0

        for b in range(1, int(bins) + 1):
            i = np.logical_and(y_prob <= b / bins, y_prob > (b - 1) / bins)  # indexes for p in the current bin

            s = np.sum(i)
            if s == 0:
                continue

            m = 1 / s
            acc = m * np.sum(y_pred[i] == y_true[i])
            conf = np.mean(y_prob[i])

            prob_pred = np.hstack((prob_pred, conf))
            prob_true = np.hstack((prob_true, acc))

            ece += s/len(y_true) * np.abs(acc - conf)

        return prob_pred, prob_true, ece