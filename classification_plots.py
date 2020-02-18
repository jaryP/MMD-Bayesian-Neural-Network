import pickle
import time

import matplotlib.pyplot as plt
import torch
import torchvision
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import transforms
import random
from shutil import copy
import ANN
import DropoutNet
from base import Wrapper
from priors import Gaussian, Laplace, ScaledMixtureGaussian, Uniform
from bayesian_layers import BayesianCNNLayer, BayesianLinearLayer
import csv
from itertools import cycle
import os

font = {'font.family': 'serif',
        'axes.labelsize': 11,
        'font.size': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': [4, 3],
        'text.usetex': False,
        'font.serif': ['Times New Roman'] + plt.rcParams['font.serif']}

plt.rcParams.update(font)


def unbalanced_mean_std(x):
    m = []
    std = []

    mx = len(max(x, key=len))

    res = np.zeros((len(x), mx), dtype=np.float)

    for i, r in enumerate(x):
        res[i, :len(r)] = r

    res = np.asarray(res)

    for i in res.T:
        v = [_v for _v in i if _v > 0]
        m.append(np.mean(v))
        std.append(np.std(v))

    return np.asarray(m), np.asarray(std)


def get_dataset(name, batch_size, dev_split, resize=None, train_subset=0):
    if name in ['fMNIST', 'MNIST']:

        tr = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

        if resize is not None:
            tr.insert(0, transforms.Resize(resize))

        if name == "fMNIST":
            tr.append(torch.flatten)

        image_transform = transforms.Compose(tr)

        train_split = datasets.MNIST('./Datasets/MNIST', train=True, download=True,
                                     transform=image_transform)
        test_split = datasets.MNIST('./Datasets/MNIST', train=False, download=True,
                                    transform=image_transform)

        sample = train_split[0][0]
        classes = 10

    if name == 'CIFAR10':
        tr = [transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]

        if resize is not None:
            tr.insert(0, transforms.Resize(resize))

        transform = transforms.Compose(tr)

        train_split = torchvision.datasets.CIFAR10(root='./Datasets/CIFAR10', train=True,
                                                   download=True, transform=transform)

        test_split = torchvision.datasets.CIFAR10(root='./Datasets/CIFAR10', train=False,
                                                  download=True, transform=transform)
        classes = 10
        sample = train_split[0][0]

    if name == 'CIFAR100':
        tr = [transforms.ToTensor(),
              transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]

        if resize is not None:
            tr.insert(0, transforms.Resize(resize))

        transform = transforms.Compose(tr)

        train_split = torchvision.datasets.CIFAR100(root='./Datasets/CIFAR100', train=True,
                                                    download=True, transform=transform)

        test_split = torchvision.datasets.CIFAR100(root='./Datasets/CIFAR100', train=False,
                                                   download=True, transform=transform)
        classes = 100
        sample = train_split[0][0]

    sampler = None
    shuffle = True

    if train_subset > 0:
        # print(train_split.data[0][0][0])

        train_size = int(train_subset * len(train_split))
        idx = np.random.choice(train_split.data.shape[0], train_size, replace=False)
        # idx = sorted(idx)
        # idx = np.random.randint(train_split.data.shape[0], size=train_size)
        # print(idx)|
        # train_split.data = train_split.data[idx]
        sampler = SubsetRandomSampler(idx)
        shuffle = False
        # print(train_split.data.shape)
        #
        # print(train_split.data[0][0][0])

    # if dev_split > 0:
    #     train_size = int((1 - dev_split) * len(train_split))
    #     test_size = len(train_split) - train_size
    #
    #     train_split, dev_split = torch.utils.data.random_split(train_split, [train_size, test_size])
    #
    #     train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
    #
    #     test_loader = torch.utils.data.DataLoader(dev_split, batch_size=batch_size, shuffle=False)
    #
    # else:

    train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

    test_loader = torch.utils.data.DataLoader(test_split, batch_size=batch_size, shuffle=False)

    return sample, classes, train_loader, test_loader


def plot_test(exps, tests, path, type='uncertainty', whist=1.5):
    tot_scores = []

    for d, t in zip(exps, tests):
        fig_corr, ax_corr = plt.subplots(nrows=1)
        fig_wro, ax_wro = plt.subplots(nrows=1)

        cm, cv = [], []
        wm, wv = [], []

        scores_keep = []
        scores = []

        if t is None:
            continue

        threshold = np.inf

        for i in range(len(t[0])):

            c, w = t[0][i], t[1][i]
            c = np.asarray(c)
            w = np.asarray(w)

            score = len(c) / (len(c) + len(w))

            # print('Score', score)

            if i == 0:
                threshold = np.quantile(c, 0.75) + whist * (np.quantile(c, 0.75) - np.quantile(c, 0.25))

            cm.append(c)
            wm.append(w)

            keep_score = score
            if i > 0:
                keep_c = [k for k in c if k > threshold]
                keep_w = [k for k in w if k > threshold]
                keep_score = len(keep_c) / (len(keep_w) + len(keep_c))

                scores.append(score)
                scores_keep.append(keep_score)
                # print('old_score/new_score', keep_score / score, '\n',
                #       'Percentage corrected classified keep', len(keep_c) / len(c), '\n',
                #       'Percentage wrongly classified keep', len(keep_w) / len(w), '\n',
                #       'Keep score', len(keep_c) / (len(keep_w) + len(keep_c)))
                # print()

            scores.append(score)
            scores_keep.append(keep_score)

        bp = ax_corr.boxplot(cm, showfliers=False, whis=whist)
        bp = ax_wro.boxplot(wm, showfliers=False, whis=whist)

        ax_wro.grid(True, color="0.9", linestyle='--', linewidth=1)
        ax_wro.set_xlabel(r'$\epsilon$', size=50)
        ax_wro.set_xticklabels(ANN.Trainer.epsilons, fontsize=8)

        ax_corr.grid(True, color="0.9", linestyle='--', linewidth=1)
        ax_corr.set_xlabel(r'$\epsilon$', size=50)
        ax_corr.set_xticklabels(ANN.Trainer.epsilons, fontsize=8)

        fig_corr.savefig(os.path.join(path, "{}_{}_correct.pdf".format(d['label'], type)),
                         bbox_inches='tight', pad_inches=0)

        fig_wro.savefig(os.path.join(path, "{}_{}_wrong.pdf".format(d['label'], type)),
                        bbox_inches='tight', pad_inches=0)

        # ax_score.plot(range(len(scores)), scores_keep, linestyle='--',
        #               label='Keep score', c=d['color'], linewidth=2)
        #
        # ax_score.plot(range(len(scores)), scores, linestyle='-',
        #               label='Original score', c=d['color'], linewidth=2)
        #
        # fig_scores.savefig(os.path.join(path, "{}_{}_scores.pdf".format(d['label'], type)),
        #                    bbox_inches='tight', pad_inches=0)
        #
        # fig_scores.legend()#, prop={'size': 25})

        plt.close('all')

        tot_scores.append((scores, scores_keep))
        return scores, scores_keep


def uncertainty_test(exps, tests, whists=None):
    tot_scores = []

    # for t in tests:

    # cm, cv = [], []
    # wm, wv = [], []
    #
    # scores_keep = []
    # scores = []
    # if t is None:
    #     continue

    threshold = np.inf
    if whists is None:
        whists = [None, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5]

    discard = np.zeros((len(tests[0]), len(whists)))
    keep = np.zeros((len(tests[0]), len(whists)))
    scores = np.zeros((len(tests[0]), len(whists)))

    for eps in range(len(tests[0])):

        for j, whist in enumerate(whists):
            c, w = tests[0][eps], tests[1][eps]

            # w = np.asarray(w)

            # score = len(c) / (len(c) + len(w))

            # if eps == 0:
            # c = np.asarray(c)

            if whist is None:
                threshold = 0
            else:
                threshold = np.quantile(c, 0.75) + whist * (np.quantile(c, 0.75) - np.quantile(c, 0.25))
            # print(threshold)
            # else:
            all = c + w

            # cm.append(c)
            # wm.append(w)
            # print(all)

            _disc = [k for k in all if k > threshold]
            _keep = [k for k in all if k <= threshold]

            discard[eps, j] = len(_disc)
            keep[eps, j] = len(_keep)

            keep_c = [k for k in c if k <= threshold]
            keep_w = [k for k in w if k <= threshold]
            div = len(keep_w) + len(keep_c)

            if div == 0:
                keep_score = 0
            else:
                keep_score = len(keep_c) / div

            scores[eps, j] = keep_score
            # keep_score = score
            # if i > 0:
            # keep_score = len(keep_c) / (len(keep_w) + len(keep_c))

            # scores.append(score)
            # scores_keep.append(keep_score)
            # print('old_score/new_score', keep_score / score, '\n',
            #       'Percentage corrected classified keep', len(keep_c) / len(c), '\n',
            #       'Percentage wrongly classified keep', len(keep_w) / len(w), '\n',
            #       'Keep score', len(keep_c) / (len(keep_w) + len(keep_c)))
            # print()

            # scores.append(score)
            # scores_keep.append(keep_score)

    # tot_scores.append((scores, scores_keep))
    return keep, discard, scores


def main(experiment):
    import sklearn.metrics as metrics
    import tqdm as tqdm

    import BBB
    import BMMD

    import os
    import numpy as np

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Optimizers = ['sgd', 'adam', 'rmsprop']
    NetworkTypes = ['bbb', 'mmd', 'normal', 'dropout']
    Datasets = ['fMNIST', 'MNIST', 'CIFAR10', 'CIFAR100']
    PosteriorType = ['layers', 'neurons', 'weights', 'multiplicative']
    LrScheduler = ['step', 'exponential', 'plateau']
    Priors = ['gaussian', 'laplace', 'uniform', 'scaledGaussian']

    linestyles = ['--', '-.', '-', ':']

    linestyles_cicle = cycle(linestyles)

    experiments_results = []
    all_fgsm_results = []
    all_fgsm_entropy_results = []
    all_reliability = []
    conf_matrices = []

    hists = []
    all_ws = []

    experiments_path = experiment['experiments_path']

    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)

    # input(experiments_path)

    experiments = experiment['experiments']

    cm = plt.get_cmap('tab20')
    cm = iter(cm.colors)

    # print(cm.colors)
    for data in experiments:
        print(data['save_path'], data['exp_name'])

        if data.get('skip', False):
            continue

        batch_size = data.get('batch_size', 64)
        lr = data.get('lr', 1e-3)
        topology = data['topology']
        weights_mu_init = data.get('mu_init', None)
        weights_rho_init = data.get('rho_init', None)
        optimizer = data.get('optimizer', 'adam').lower()
        dataset = data["dataset"]
        network = data["network_type"].lower()
        # experiments = data.get('experiments', 1)
        seeds = data.get('experiments_seeds', [0])
        device = 'cuda' if torch.cuda.is_available() and data.get('use_cuda', True) else 'cpu'
        save_path = data['save_path']
        load_path = data.get('load_path', save_path)

        loss_weights = data.get('loss_weights', {})

        epochs = data['epochs']
        train_samples = data.get('train_samples', 2)
        test_samples = data.get('test_samples', 2)
        exp_name = data['exp_name']

        if 'label' not in data:
            data['label'] = exp_name

        if 'linestyle' not in data:
            data['linestyle'] = next(linestyles_cicle)

        save = data['save']
        load = data['load']
        dev_split = data.get('dev_split', 0)
        local_trick = data.get('local_trick', False)
        # label = data.get("label", network)
        network_parameters = data.get('network_parameters', {})
        posterior_type = data.get('posterior_type', 'weights')
        moment = data.get('moment', 0)
        early_stopping_tolerance = data.get('early_stopping_tolerance', 5)
        resize = data.get('resize', None)
        weight_decay = data.get('weight_decay', 0)
        lr_scheduler = data.get('lr_scheduler', None)

        train_subset = data.get('train_subset', 0)
        # PRIORS

        # prior = None
        # data_prior = data.get('prior')
        #
        # if data_prior:
        #     t = data_prior.get('type')
        #     if t not in Priors:
        #         raise ValueError('Supported priors', list(Priors))
        #     else:
        #         if t == 'gaussian':
        #             prior = Gaussian(data_prior['mu'], data_prior['sigma'])
        #         elif t == 'laplace':
        #             prior = Laplace(data_prior['mu'], data_prior['scale'])
        #         elif t == 'scaledGaussian':
        #             prior = ScaledMixtureGaussian(pi=data_prior['phi'], mu1=data_prior['mu1'], s1=data_prior['sigma1'],
        #                                           mu2=data_prior['mu2'], s2=data_prior['sigma2'])
        #         elif t == 'uniform':
        #             a, b = data_prior['a'], data_prior['b']
        #             if network == 'bbb':
        #                 a = torch.tensor([float(a)])
        #                 b = torch.tensor([float(b)])
        #             prior = Uniform(a=a, b=b)

        # if epochs < 0:
        #     raise ValueError('The number of epoch should be > 0')

        if train_subset < 0 or train_subset > 1:
            raise ValueError('The train subset to sample needs be a percentage (> 0 and < 1)')

        if isinstance(seeds, int):
            seeds = [seeds]

        # if posterior_type not in PosteriorType:
        #     raise ValueError('Supported posterior_type', PosteriorType)

        if network not in NetworkTypes:
            raise ValueError('Supported networks', NetworkTypes)
        else:
            if network == 'mmd':
                base_model = BMMD.BMMD
                trainer = BMMD.Trainer

                if 'color' not in data:
                    data['color'] = 'red'

            elif network == 'bbb':
                base_model = BBB.BBB
                trainer = BBB.Trainer

                if 'color' not in data:
                    data['color'] = 'green'

            elif network == 'normal':
                base_model = ANN.ANN
                trainer = ANN.Trainer

                if 'color' not in data:
                    data['color'] = 'blue'

            elif network == 'dropout':
                base_model = DropoutNet.Dropnet
                trainer = DropoutNet.Trainer

                if 'color' not in data:
                    data['color'] = 'k'

        if data["color"] == "":
            data["color"] = next(cm)

        run_results = []
        _fgsm = []
        _fgsm_entropy = []
        local_reliability = []

        for e, seed in tqdm.tqdm(enumerate(seeds), total=len(seeds)):

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            if dataset not in Datasets:
                raise ValueError('Supported datasets {}, given {}'.format(Datasets, dataset))
            else:
                sample, classes, train_loader, test_loader = get_dataset(dataset, batch_size,
                                                                         dev_split, resize, train_subset)

            model = base_model(prior=None, mu_init=weights_mu_init, device=device,
                               rho_init=weights_rho_init, topology=topology, classes=classes, local_trick=local_trick,
                               sample=sample, **network_parameters, posterior_type=posterior_type)

            model.to(device)

            current_path = os.path.join(save_path, exp_name)  # , str(e))
            current_load_path = os.path.join(load_path, exp_name)

            results_path = os.path.join(current_path, 'results_{}.data'.format(e))
            best_path = os.path.join(current_path, 'best_model_{}.data'.format(e))

            if save and not os.path.exists(current_path):
                os.makedirs(current_path)

            if load and not os.path.exists(current_load_path):
                os.makedirs(current_load_path)

            if os.path.exists(os.path.join(current_load_path, 'results_{}.data'.format(e))):
                results = torch.load(results_path)
            else:
                continue

            t = trainer(model, train_loader, test_loader, None)
            run_results.append(results)

            t.model.load_state_dict(torch.load(best_path))
            run_results.append(results)

            # Reliability diagram and ECE

            # if os.path.exists(os.path.join(current_path, "{}_temp_scaling.data".format(e))):
            with open(os.path.join(current_path, "{}_temp_scaling.data".format(e)), "rb") as f:
                r = pickle.load(f)

            local_reliability.append(r)

            # Confusion matrix
            plt.close('all')
            true_class, pred_class = t.test_evaluation(samples=test_samples)
            conf_matrix = confusion_matrix(true_class, pred_class)
            plt.matshow(conf_matrix)
            plt.savefig(os.path.join(current_path, "{}_conf_matrix.pdf".format(e)),
                        bbox_inches='tight')

            fgsm_results = None
            fgsm_entropy_results = None

            if network != 'normal':

                # entropy_path = os.path.join(current_path, 'fgsm_entropy_{}.data'.format(e))
                fgsm_path = os.path.join(current_path, 'fgsm_{}.data'.format(e))

                if os.path.exists(fgsm_path):
                    with open(fgsm_path, "rb") as f:
                        fgsm_results, fgsm_entropy_results = pickle.load(f)

                # if fgsm_results is None:
                #     r = t.fgsm_test(samples=test_samples)
                #     fgsm_results, fgsm_entropy_results = r
                #
                #     if save:
                #         with open(fgsm_path, "wb") as output_file:
                #             pickle.dump(r, output_file)

                # Variance matrix
                # plt.close('all')

            _fgsm_entropy.append(fgsm_entropy_results)
            _fgsm.append(fgsm_results)

            ws = []
            to_hist = []

            if network in ['bbb', 'mmd']:

                labels = []
                for layer in t.model.features:
                    wc = []
                    if isinstance(layer, (BayesianLinearLayer, BayesianCNNLayer)):
                        labels.append('CNN' if isinstance(layer, BayesianCNNLayer) else 'Linear')

                        w = layer.w
                        b = layer.b

                        mean = np.abs(w.mu.detach().cpu().numpy())
                        std = w.sigma.detach().cpu().numpy()
                        snr = mean / std
                        to_hist.append(np.reshape(snr, -1))
                        wc.extend(np.reshape(w.weights.detach().cpu().numpy(), -1))
                        if b is not None:
                            mean = np.abs(b.mu.detach().cpu().numpy())
                            std = b.sigma.detach().cpu().numpy()
                            snr = mean / std
                            to_hist.append(np.reshape(snr, -1))

                            wc.extend(np.reshape(b.weights.detach().cpu().numpy(), -1))

                        ws.append(wc)

                to_hist = np.concatenate(to_hist)
                hists.append(to_hist)
                all_ws.append(ws)

                plt.close('all')

                fig, ax = plt.subplots()

                for l, w in enumerate(ws):
                    offset = np.abs(np.min(w) - np.max(w)) * 0.1
                    xs = np.linspace(np.min(w) - offset, np.max(w) + offset, 200)
                    density = gaussian_kde(w)
                    density._compute_covariance()
                    plt.plot(xs, density(xs))#, label="Layer #{}: {}".format(l + 1, labels[l]), linewidth=1)

                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(True, alpha=0.2)
                plt.ylabel('frequency')

                fig.savefig(os.path.join(current_path, "{}_{}_posterior.pdf".format(e, network)),
                            bbox_inches='tight', pad_inches=0)
                plt.close(fig)

        all_reliability.append(local_reliability)

        all_fgsm_entropy_results.append(_fgsm_entropy)
        all_fgsm_results.append(_fgsm)
        experiments_results.append(run_results)

    all_reliability = np.asarray(all_reliability)
    experiments = [e for e in experiments if not e.get('skip', False)]

    plt.close('all')

    threshold_tests = os.path.join(experiments_path, 'threshold_results')
    whists = [None, 0, 0.1, .2, 0.4, 0.6, 0.8, 1, 1.5, 2]

    for j, d in enumerate(experiments):
        # for each seed

        all_entropy = []
        all_unc = []

        _threshold_tests = os.path.join(threshold_tests, '{}'.format(d['label'].replace('_', ' ')))

        if not os.path.exists(_threshold_tests):
            os.makedirs(_threshold_tests)

        for i in range(len(all_fgsm_entropy_results[0])):

            if d['network_type'] == 'normal':
                continue
            threshold_results = []

            # _threshold_tests = os.path.join(threshold_tests, '{}_{}'.format(i, d['label'].replace('_', ' ')))


            _entropy = all_fgsm_entropy_results[j][i]
            _uncertainty = all_fgsm_results[j][i]

            # figsize = font['figure.figsize']
            # _figsize = tuple(map(lambda x: x * 4, figsize))

            # f, axs = plt.subplots(nrows=2, ncols=3, figsize=_figsize)

            # whists_l = [i if i is not None else 'Baseline' for i in whists]
            # x = range(len(whists))

            all_entropy.append(uncertainty_test(experiments, _entropy, whists=whists))
            all_unc.append(uncertainty_test(experiments, _uncertainty, whists=whists))

            for vi, vals in enumerate([_entropy, _uncertainty]):
                res = uncertainty_test(experiments, vals, whists=whists)
                threshold_results.append(res)

            #     for k, r in enumerate(res):
            #         axs[vi, k].matshow(r)
            #         axs[vi, k].set_yticklabels([''] + Wrapper.epsilons)
            #         axs[vi, k].set_xticks(x)
            #         axs[vi, k].set_xticklabels(whists_l, rotation=45)
            #
            # f.savefig(os.path.join(_threshold_tests, "heatmaps_{}_{}_wrong.pdf".format(d['label'], i)),
            #           bbox_inches='tight', pad_inches=0)

        eps = Wrapper.epsilons
        whists = whists[1:]
        x = range(len(whists))

        all_entropy = np.asarray(all_entropy)
        all_entropy_m = np.mean(all_entropy, 0)
        all_entropy_std = np.std(all_entropy, 0)

        all_unc = np.asarray(all_unc)
        all_unc_m = np.mean(all_unc, 0)
        all_unc_std = np.std(all_unc, 0)

        for e in range(1, len(eps)):
            ep = eps[e]

            # entr_disc = threshold_results[0][1][e, 1:]
            entr_disc = all_entropy_m[1, e, 1:]
            entr_disc_std = all_entropy_std[1, e, 1:]

            # unc_disc = threshold_results[1][1][e, 1:]
            unc_disc = all_unc_m[1, e, 1:]
            unc_disc_std = all_unc_std[1, e, 1:]

            # entr_score = threshold_results[0][-1][e, 1:] * 100
            # unsc_score = threshold_results[1][-1][e, 1:] * 100

            entr_score = all_entropy[:, 2, e, 1:] * 100
            unsc_score = all_unc[:, 2, e, 1:] * 100

            score_diff_m = np.mean(unsc_score - entr_score, 0)
            score_diff_std = np.std(unsc_score - entr_score, 0)
            # diff = unsc_score - entr_score

            f, axs = plt.subplots(nrows=1, ncols=1)  # , figsize=_figsize)
            axs.plot(x, score_diff_m, label='Unc. - Entropy', linestyle='-')
            axs.fill_between(x, score_diff_m - score_diff_std, score_diff_m + score_diff_std, alpha=0.1, color=d['color'])

            axs.set_xticks(x)
            axs.set_xticklabels(whists)
            axs.set_xlabel(r'$\gamma$')
            axs.set_ylabel('Unc. - Entropy %')

            # plt.fill_between(x, means - stds, means + stds, alpha=0.1, color=d['color'])

            axs.grid(True, color="0.9", linestyle='--', linewidth=1)
            f.savefig(os.path.join(_threshold_tests, "{}_score_difference.pdf".format(ep)),
                      bbox_inches='tight', pad_inches=0)

            f, axs = plt.subplots(nrows=1, ncols=1)  # , figsize=figsize)
            axs.plot(x, entr_disc, label='Entropy', linestyle='-')
            axs.fill_between(x, entr_disc - entr_disc_std, entr_disc + entr_disc_std, alpha=0.1, color=d['color'])

            axs.plot(x, unc_disc, label='Unc.', linestyle='-.')
            axs.fill_between(x, unc_disc - unc_disc_std, unc_disc + unc_disc_std, alpha=0.1, color=d['color'])

            axs.set_xticks(x)
            axs.set_xticklabels(whists)
            axs.set_xlabel(r'$\gamma$')
            axs.set_ylabel('discarded images')

            axs.grid(True, color="0.9", linestyle='--', linewidth=1)
            f.legend(bbox_to_anchor=(0.9, 0.9))  # loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0))
            f.savefig(os.path.join(_threshold_tests, "{}_disc.pdf".format(ep)),
                      bbox_inches='tight', pad_inches=0)

            f, axs = plt.subplots(nrows=1, ncols=1)  # , figsize=figsize)

            axs.plot(x, entr_score, label='Entropy', linestyle='-')
            axs.plot(x, unsc_score, label='Unc.', linestyle='-.')

            axs.set_xticks(x)
            axs.set_xticklabels(whists)
            axs.set_xlabel(r'$\gamma$')
            axs.set_ylabel('accuracy %')

            axs.grid(True, color="0.9", linestyle='--', linewidth=1)
            f.legend(bbox_to_anchor=(0.9, 0.9))  # loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0))
            f.savefig(os.path.join(_threshold_tests, "{}_score.pdf".format(ep)),
                      bbox_inches='tight', pad_inches=0)

            plt.close('all')

    # Accuracy plot
    plt.figure(0)

    for i in range(len(experiments)):
        d = experiments[i]
        r = experiments_results[i]

        res = [i['test_results'] for i in r]

        means, stds = unbalanced_mean_std(res)

        plt.plot(range(len(means)), means, linestyle=d['linestyle'],
                 label=d.get('label', d['network_type']), c=d['color'], linewidth=2)

        plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.1, color=d['color'])

    plt.legend()

    plt.ylabel("Test score (%)")
    plt.xlabel("Epochs")

    print(os.path.join(experiments_path, "score.pdf"))
    plt.savefig(os.path.join(experiments_path, "score.pdf"), bbox_inches='tight', pad_inches=0)
    plt.close('all')

    # Results file writing

    with open(os.path.join(experiments_path, 'results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["exp_name", "network", "max_score", "max_score_epoch", "ece", "scaled_ece"])

        for i in range(len(experiments)):
            d = experiments[i]
            r = experiments_results[i]

            ece = all_reliability[i, :, 0] * 100

            res = [i['test_results'] for i in r]
            res, res_std = unbalanced_mean_std(res)

            res_i = np.argmax(res)

            writer.writerow([d.get('label'), d.get('network_type'),
                             '{} +- {}'.format(res[res_i] * 100, res_std[res_i] * 100),
                             res_i,
                             '{} +- {}'.format(np.mean(ece), np.std(ece)),
                             '{} +- {}'.format(np.mean(ece), np.std(ece))])

    return experiments_results


if __name__ == '__main__':
    import json
    import sys
    import numpy as np

    args = sys.argv[1:]

    with open(args[0], "r") as read_file:
        experiments = json.load(read_file)

    results = main(experiments)
