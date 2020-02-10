from abc import ABC, abstractmethod
from copy import copy
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.optimize import minimize
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.distributions import Exponential
from tqdm import tqdm

from bayesian_layers import BayesianCNNLayer, BayesianLinearLayer


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
                zip(np.random.choice(idx, int(ln * self.percentage)), np.random.choice(idx, int(ln * self.percentage)))

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
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def log_gaussian_loss(out_dim):
    def loss_function(x, y, sigma):
        exponent = -0.5 * (x - y) ** 2 / sigma ** 2
        log_coeff = -torch.log(sigma + 1e-12) - 0.5 * np.log(2 * np.pi)
        return -(log_coeff + exponent).sum()

    return loss_function


def cross_entropy_loss(reduction):
    def loss_function(x, y):
        _x = F.log_softmax(x, -1)
        if _x.dim() == 3:
            _x = _x.mean(0)
        return F.nll_loss(_x, y, reduction=reduction)

    return loss_function


# def epistemic_aleatoric_variance(x):
#
#     if x.dim() == 2:
#         x = x.unsqueeze(0)
#
#     p = torch.softmax(x, -1)
#     p_hat = torch.mean(p, 0)
#     p = p.detach().cpu().numpy()
#     p_hat = p_hat.detach().cpu().numpy()
#
#     t = p.shape[0]
#     classes = p.shape[-1]
#
#     vars = []
#
#     for _p, _p_hat in zip(p, p_hat):
#         aleatoric = np.zeros((classes, classes))
#         epistemic = np.zeros((classes, classes))
#
#         for ip, ip_hat in zip(_p, _p_hat):
#             aleatoric += np.diag(ip) - np.outer(ip, ip)
#             d = ip - ip_hat
#             epistemic += np.outer(d, d)
#
#         aleatoric /= t
#         epistemic /= t
#
#         vars.append((aleatoric, epistemic))
#
#     return np.asarray(vars)


def epistemic_aleatoric_uncertainty(x):
    if x.dim() == 2:
        x = x.unsqueeze(0)

    p = torch.softmax(x, 2)
    p_hat = torch.mean(p, 0)

    p = p.detach().cpu().numpy()
    p = np.transpose(p, (1, 0, 2))

    p_hat = p_hat.detach().cpu().numpy()

    t = p.shape[1]
    classes = p.shape[-1]

    determinants = []
    variances = []

    mn = 1 / classes ** classes
    mx = mn * (2 ** (classes - 1))

    for _bi in range(p.shape[0]):
        _bp = p[_bi]
        _bp_hat = p_hat[_bi]

        al = np.zeros((classes, classes))
        ep = np.zeros((classes, classes))

        for i in range(t):
            _p = _bp[i]
            aleatoric = np.diag(_p) - np.outer(_p, _p)
            al += aleatoric
            d = _p - _bp_hat
            epistemic = np.outer(d, d)
            ep += epistemic

        al /= t
        ep /= t

        var = al + ep

        variances.append(var)

        det = np.linalg.det(var + (np.eye(classes) / classes))

        det = (det - mn) / (mx - mn)
        determinants.append(det)


    determinants = np.asarray(determinants)
    variances = np.asarray(variances)

    return determinants, variances


def compute_entropy(preds, sum=True):
    l = torch.log10(preds + 1e-12) * preds
    if sum:
        return -torch.sum(l, 1)
    else:
        return -l


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

        elif isinstance(i, str) and i.lower() == 'sigmoid':
            l = torch.nn.Sigmoid()

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
                                        prior=prior, divergence=divergence, local_rep_trick=local_trick, use_bias=bias,
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
            l = torch.nn.Linear(prev, i, bias=bias)
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
    epsilons = [0, 0.001, 0.005, 0.01, 0.05]  # , .2, .5]
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

    def test_evaluation(self, samples, temperature=1, **kwargs):

        test_pred = []
        test_true = []

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(self.test_data), leave=False, total=len(self.test_data)):
                x = x.to(self.device)
                y = y.to(self.device)

                test_true.extend(y.tolist())

                out = self.model.eval_forward(x.to(self.device), samples=samples)
                out = torch.mul(out, temperature)

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

                    a, _ = epistemic_aleatoric_uncertainty(out)
                    H.extend(a)

                    if out.dim() > 2:
                        out = out.mean(0)

                    out = torch.softmax(out, -1)
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

    def fgsm_test(self, samples=1):

        correctly_predicted = []
        wrongly_predicted = []

        self.model.eval()
        loss = cross_entropy_loss('mean')

        for eps in tqdm(self.epsilons, desc='Attack test', leave=False):

            H = []
            pred_label = []
            true_label = []

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

                with torch.no_grad():
                    perturbed_data = fgsm_attack(x, eps)

                    out = self.model.eval_forward(perturbed_data, samples=samples)

                    a, _ = epistemic_aleatoric_uncertainty(out)
                    H.extend(a)

                    out = torch.softmax(out, -1)
                    if out.dim() > 2:
                        out = out.mean(0)

                    pred_label.extend(out.argmax(dim=-1).tolist())

                    # top_score, top_label = torch.topk(out, 2)

            # H = np.mean(-np.log(H + 1e-12))

            # print(np.mean([H[i] for i in range(len(true_label)) if true_label[i] == pred_label[i]]))
            # print(np.mean([H[i] for i in range(len(true_label)) if true_label[i] != pred_label[i]]))

            _correctly_predicted = []
            _wrongly_predicted = []

            for i in range(len(true_label)):
                if true_label[i] == pred_label[i]:
                    _correctly_predicted.append(H[i])
                else:
                    _wrongly_predicted.append(H[i])

            correctly_predicted.append(np.mean(_correctly_predicted))
            wrongly_predicted.append(np.mean(_wrongly_predicted))

        # correctly_predicted, wrongly_predicted = np.asarray(correctly_predicted), np.asarray(wrongly_predicted)

        return correctly_predicted, wrongly_predicted

    def reliability_diagram(self, samples=1, bins=15, scaling=1, **kwargs):

        y_prob = []
        y_true = []
        y_pred = []

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_data):
                y_true.extend(y.tolist())
                x = x.to(self.device)

                out = self.model.eval_forward(x.to(self.device), samples=samples)

                if hasattr(out, '__call__') and hasattr(out, 'sample'):
                    scaling = out.sample(out.size()[-1])

                out = torch.div(out, scaling)

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
        nll = -np.sum(np.log(y_prob))

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

            ece += s / len(y_true) * np.abs(acc - conf)

        return prob_pred, prob_true, ece, nll

    def total_variance(self, samples=1, **kwargs):

        M = []

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_data):
                x = x.to(self.device)
                out = self.model.eval_forward(x, samples=samples)

                _, m = epistemic_aleatoric_uncertainty(out)

                M.extend(m)

        M = np.asarray(M)
        M = M.mean(0)

        return M

    def temperature_scaling(self, samples=1, **kwargs):
        #  Based on https://github.com/gpleiss/temperature_scaling/
        temperature = nn.Parameter(torch.ones(10, device=self.device) * 1.5, requires_grad=True)
        # temperature = Exponential(temp)

        self.model.eval()

        nll_criterion = nn.CrossEntropyLoss()

        _, _, before_ece, before_nll = self.reliability_diagram(samples=samples)

        optimizer = optim.LBFGS([temperature], lr=0.05, max_iter=20)

        def eval():

            loss = 0

            # with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(self.test_data), leave=False, total=len(self.test_data)):
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    out = self.model.eval_forward(x.to(self.device), samples=samples)

                out = torch.div(out, temperature)

                if out.dim() > 2:
                    out = out.mean(0)

                loss += nll_criterion(out, y)

            loss = torch.div(loss, i)
            loss.backward()

            #
            #         logits_list.extend(out)
            #         labels_list.extend(y)
            #
            #         # break
            #
            # logits = torch.stack(logits_list)  # .to(self.device)
            # labels = torch.stack(labels_list)  # .to(self.device)

            # print(logits.shape)
            # logits = torch.div(logits, temperature)

            # _, _, ece, _ = self.reliability_diagram(samples=samples, scaling=temperature)

            # print(logits.size())
            # if hasattr(temperature, 'sample'):
            #     scaling = temperature.rsample()
            #     input(scaling.sample())
            # else:
            #     scaling = temperature

            # print(scaling)

            # loss = nll_criterion(logits, labels)
            # print(loss.item(), temperature.tolist())
            # loss.backward()

            return loss

        optimizer.step(eval)

        a, b, ece, nll = self.reliability_diagram(samples=samples, scaling=temperature)

        test_true, test_pred = self.test_evaluation(samples=samples)
        f1 = metrics.f1_score(test_true, test_pred, average='micro')

        print(temperature.tolist(), before_ece * 100, ece * 100, f1)

        return a, b, ece, nll


if __name__ == '__main__':

    def prova(inp):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)

        p = inp
        p_hat = torch.mean(p, 0)

        p = p.detach().cpu().numpy()
        p = np.transpose(p, (1, 0, 2))

        p_hat = p_hat.detach().cpu().numpy()

        t = p.shape[1]
        classes = p.shape[-1]

        determinants = []
        variances = []

        mn = 1/classes**classes
        mx = mn * (2**(classes-1))

        for _bi in range(p.shape[0]):
            _bp = p[_bi]
            _bp_hat = p_hat[_bi]

            al = np.zeros((classes, classes))
            ep = np.zeros((classes, classes))

            for i in range(t):
                _p = _bp[i]
                aleatoric = np.diag(_p) - np.outer(_p, _p)
                al += aleatoric
                d = _p - _bp_hat
                epistemic = np.outer(d, d)
                ep += epistemic

            al /= t
            ep /= t

            print(al+ep)
            variances.append(al+ep+(np.eye(classes) / classes))

            det = np.linalg.det(variances[-1])

            det = (det - mn) / (mx-mn)
            determinants.append(det)

        determinants = np.asarray(determinants)
        variances = np.asarray(variances)

        return determinants, variances

    x = torch.tensor(
        [
            [
                [.25, .25, .25, .25],
            ]
        ],
        dtype=torch.float
    ).permute(1, 0, 2)

    print(x.shape)
    d = (1/x.shape[-1])**x.shape[-1]
    print(d)

    a, b = prova(x)
    print(a)

    # a, _ = epistemic_aleatoric_uncertainty(x)
    # print(a)

    # print(len(list(permutations( [[1, 0, 0, 0 ],
    #             [0, 1, 0, 0],
    #             [0, 0, 1, 0],
    #             [0, 0, 0, 1]]))))