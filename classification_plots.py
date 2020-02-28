import pickle

import matplotlib.pyplot as plt
import torch
import torchvision
from scipy.interpolate import make_interp_spline
from scipy.stats import gaussian_kde
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import transforms
import random
import json
import sys
import numpy as np
import ANN
import DropoutNet
from base import Wrapper
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
        'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
        'figure.autolayout': True}

plt.rcParams.update(font)


def smoother(x, space, points=300):
    xnew = np.linspace(np.min(space), np.max(space), points)
    spl = make_interp_spline(range(len(x)), x, k=3)
    power_smooth = spl(xnew)

    return xnew, power_smooth


def moving_average(x, n=3):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if n % 2 == 0:
            a, b = i - (n - 1) // 2, i + (n - 1) // 2 + 2
        else:
            a, b = i - (n - 1) // 2, i + (n - 1) // 2 + 1

        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


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
        train_size = int(train_subset * len(train_split))
        idx = np.random.choice(train_split.data.shape[0], train_size, replace=False)
        sampler = SubsetRandomSampler(idx)
        shuffle = False

    train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

    test_loader = torch.utils.data.DataLoader(test_split, batch_size=batch_size, shuffle=False)

    return sample, classes, train_loader, test_loader


def plot_test(exps, tests, path, type='uncertainty', whist=2):
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


        plt.close('all')

        tot_scores.append((scores, scores_keep))
        return scores, scores_keep


def uncertainty_test(exps, tests, whists=None):

    threshold = np.inf
    if whists is None:
        whists = [None, 0.25, 0.5, 0.75, 1, 2, 2, 2.5]

    discard = np.zeros((len(tests[0]), len(whists)))
    keep = np.zeros((len(tests[0]), len(whists)))
    scores = np.zeros((len(tests[0]), len(whists)))

    for eps in range(len(tests[0])):

        for j, whist in enumerate(whists):
            c, w = tests[0][eps], tests[1][eps]

            if whist is None:
                threshold = 0
            else:
                threshold = np.quantile(c, 0.75) + whist * (np.quantile(c, 0.75) - np.quantile(c, 0.25))
            # else:
            all = c + w

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

    return keep, discard, scores


def main(experiment):
    import tqdm as tqdm

    import BBB
    import BMMD

    import os
    import numpy as np

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    NetworkTypes = ['bbb', 'mmd', 'normal', 'dropout']
    Datasets = ['fMNIST', 'MNIST', 'CIFAR10', 'CIFAR100']

    linestyles = ['--', '-.', '-', ':']

    linestyles_cicle = cycle(linestyles)

    experiments_results = []
    all_fgsm_results = []
    all_fgsm_entropy_results = []
    all_reliability = []
    all_bars = []

    hists = []
    all_ws = []

    experiments_path = experiment['experiments_path']

    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)

    experiments = experiment['experiments']

    cm = plt.get_cmap('tab20')
    cm = iter(cm.colors)

    for data in experiments:
        print(data['save_path'], data['exp_name'])

        if data.get('skip', False):
            continue

        batch_size = data.get('batch_size', 64)
        topology = data['topology']
        weights_mu_init = data.get('mu_init', None)
        weights_rho_init = data.get('rho_init', None)
        dataset = data["dataset"]
        network = data["network_type"].lower()
        # experiments = data.get('experiments', 1)
        seeds = data.get('experiments_seeds', [0])
        device = 'cuda' if torch.cuda.is_available() and data.get('use_cuda', True) else 'cpu'
        save_path = data['save_path']
        load_path = data.get('load_path', save_path)

        test_samples = data.get('test_samples', 2)
        exp_name = data['exp_name']
        seeds = [0, 1, 2]

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

        if train_subset < 0 or train_subset > 1:
            raise ValueError('The train subset to sample needs be a percentage (> 0 and < 1)')

        if isinstance(seeds, int):
            seeds = [seeds]

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
        local_bars = []

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

            current_path = os.path.join(save_path, exp_name)
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

            if not os.path.exists(os.path.join(current_path, "{}_ece.data".format(e))):
                assert False

            with open(os.path.join(current_path, "{}_ece.data".format(e)), "rb") as f:
                r = pickle.load(f)

            with open(os.path.join(current_path, "{}_ece_barplot.data".format(e)), "rb") as f:
                r1 = pickle.load(f)

            local_reliability.append(r)
            local_bars.append(r1)

            fgsm_results = None
            fgsm_entropy_results = None

            if network != 'normal':

                fgsm_path = os.path.join(current_path, 'fgsm_{}.data'.format(e))

                if os.path.exists(fgsm_path):
                    with open(fgsm_path, "rb") as f:
                        fgsm_results, fgsm_entropy_results = pickle.load(f)
                else:
                    assert False

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
                fig.set_figheight(3)
                fig.set_figwidth(4)

                for l, w in enumerate(ws):
                    offset = np.abs(np.min(w) - np.max(w)) * 0.1
                    xs = np.linspace(np.min(w) - offset, np.max(w) + offset, 200)
                    density = gaussian_kde(w)
                    density._compute_covariance()
                    plt.plot(xs, density(xs))

                ax.grid(True, alpha=0.2)
                ax.set_ylabel('frequency')
                ax.margins(y=0)
                ax.set_xlim(-0.5, 0.5)

                mn, mx = ax.get_ylim()
                ax.set_ylim(mn, mx + (mx * 0.2))

                fig.savefig(os.path.join(current_path, "{}_{}_posterior.pdf".format(e, network)),
                            bbox_inches='tight', pad_inches=0)
                plt.close(fig)

        all_reliability.append(local_reliability)
        all_bars.append(local_bars)

        all_fgsm_entropy_results.append(_fgsm_entropy)
        all_fgsm_results.append(_fgsm)
        experiments_results.append(run_results)

    experiments = [e for e in experiments if not e.get('skip', False)]

    plt.close('all')

    threshold_tests = os.path.join(experiments_path, 'threshold_results')
    whists = [None, 0, 0.1, .2, 0.4, 0.6, 0.8, 1, 1.5, 2]

    for j, d in enumerate(experiments):

        if d['network_type'] == 'normal':
            continue

        all_entropy = []
        all_unc = []

        _threshold_tests = os.path.join(threshold_tests, '{}'.format(d['label'].replace('_', ' ')))

        if not os.path.exists(_threshold_tests):
            os.makedirs(_threshold_tests)

        for i in range(len(all_fgsm_entropy_results[0])):
            _entropy = all_fgsm_entropy_results[j][i]
            _uncertainty = all_fgsm_results[j][i]

            all_entropy.append(uncertainty_test(experiments, _entropy, whists=whists))
            all_unc.append(uncertainty_test(experiments, _uncertainty, whists=whists))

        eps = Wrapper.epsilons
        whists = whists[1:]
        x = range(len(whists))
        _x = x

        all_entropy = np.asarray(all_entropy)
        all_entropy_m = np.mean(all_entropy, 0)
        all_entropy_std = np.std(all_entropy, 0)

        all_unc = np.asarray(all_unc)
        all_unc_m = np.mean(all_unc, 0)
        all_unc_std = np.std(all_unc, 0)

        for e in range(1, len(eps)):
            ep = eps[e]

            entr_disc = all_entropy_m[1, e, 1:]
            entr_disc_std = all_entropy_std[1, e, 1:]

            unc_disc = all_unc_m[1, e, 1:]
            unc_disc_std = all_unc_std[1, e, 1:]

            entr_score = all_entropy[:, 2, e, 1:] * 100
            unsc_score = all_unc[:, 2, e, 1:] * 100

            score_diff_m = np.mean(unsc_score - entr_score, 0)
            score_diff_std = np.std(unsc_score - entr_score, 0)

            _, score_diff_m = smoother(score_diff_m, _x)
            x1, score_diff_std = smoother(score_diff_std, _x)

            f, axs = plt.subplots(nrows=1, ncols=1)
            f.set_figheight(3 * 1)
            f.set_figwidth(4 * 1)

            axs.plot(x1, score_diff_m, label='BCU - Entropy', linestyle='-', color='r')
            axs.fill_between(x1, score_diff_m - score_diff_std, score_diff_m + score_diff_std, alpha=0.1,
                             color='r')

            axs.set_xticks(x)
            axs.set_xticklabels(whists)

            axs.set_xlabel(r'$\gamma$')
            axs.set_ylabel('BCU - Entropy (%)')
            axs.margins(x=0)

            axs.grid(True, color="0.9", linestyle='--', linewidth=1)

            f.savefig(os.path.join(_threshold_tests, "{}_score_difference.pdf".format(ep)), )

            #######################################################################################################

            f, axs = plt.subplots(nrows=1, ncols=1)
            f.set_figheight(3 * 1)
            f.set_figwidth(4 * 1)

            x1, entr_disc = smoother(entr_disc, _x)
            _, entr_disc_std = smoother(entr_disc_std, _x)


            axs.plot(x1, entr_disc, label='Entropy', linestyle='-')
            axs.fill_between(x1, entr_disc - entr_disc_std, entr_disc + entr_disc_std, alpha=0.1)

            x1, unc_disc = smoother(unc_disc, _x)
            _, unc_disc_std = smoother(unc_disc_std, _x)

            axs.plot(x1, unc_disc, label='BCU', linestyle='-.')
            axs.fill_between(x1, unc_disc - unc_disc_std, unc_disc + unc_disc_std, alpha=0.1)

            axs.set_xticks(x)
            axs.set_xticklabels(whists)

            axs.set_xlabel(r'$\gamma$')
            axs.set_ylabel('discarded images')
            axs.margins(x=0)

            mn, mx = axs.get_ylim()
            axs.set_ylim(mn, mx+(mx*0.2))

            axs.grid(True, color="0.9", linestyle='--', linewidth=1)
            f.legend(bbox_to_anchor=(0.95, 0.95), ncol=2)

            f.savefig(os.path.join(_threshold_tests, "{}_disc.pdf".format(ep)), )

            #######################################################################################################

            f, axs = plt.subplots(nrows=1, ncols=1)
            f.set_figheight(3 * 1)
            f.set_figwidth(4 * 1)

            entr_score = all_entropy_m[2, e, 1:] * 100
            entr_std = all_entropy_std[2, e, 1:] * 100

            unsc_score = all_unc_m[2, e, 1:] * 100
            unc_std = all_unc_std[2, e, 1:] * 100

            x1, entr_score = smoother(entr_score, _x)
            _, entr_std = smoother(entr_std, _x)

            _, unsc_score = smoother(unsc_score, _x)
            _, unc_std = smoother(unc_std, _x)

            axs.plot(x1, entr_score, label='Entropy', linestyle='-')
            axs.fill_between(x1, entr_score - entr_std, entr_score + entr_std, alpha=0.1)

            axs.plot(x1, unsc_score, label='BCU', linestyle='-.')
            axs.fill_between(x1, unsc_score - unc_std, unsc_score + unc_std, alpha=0.1)

            axs.set_xticks(x)
            axs.set_xticklabels(whists)
            axs.margins(x=0)

            axs.set_xlabel(r'$\gamma$')
            axs.set_ylabel('accuracy (%)')

            axs.grid(True, color="0.9", linestyle='--', linewidth=1)

            mn, mx = axs.get_ylim()
            axs.set_ylim(mn, mx+(mx*0.2))
            f.legend(bbox_to_anchor=(0.95, 0.95), ncol=2)

            f.savefig(os.path.join(_threshold_tests, "{}_score.pdf".format(ep)), )

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

    plt.savefig(os.path.join(experiments_path, "score.pdf"), bbox_inches='tight', pad_inches=0)
    plt.close('all')

    # Results file writing

    all_reliability = np.asarray(all_reliability)
    all_bars = np.asarray(all_bars)
    eces = []
    with open(os.path.join(experiments_path, 'results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["exp_name", "network", "max_score", "max_score_epoch", "ece", "scaled_ece"])

        for i in range(len(experiments)):
            d = experiments[i]
            r = experiments_results[i]

            ece = all_reliability[i, :, 0] * 100
            scaled_ece = all_reliability[i, :, 1] * 100

            res = [i['test_results'] for i in r]
            res, res_std = unbalanced_mean_std(res)

            res_i = np.argmax(res)

            eces.append((np.mean(ece), np.std(ece)))

            writer.writerow([d.get('label'), d.get('network_type'),
                             '{} +- {}'.format(res[res_i] * 100, res_std[res_i] * 100),
                             res_i,
                             '{} +- {} ({})'.format(np.mean(ece), np.std(ece), np.min(ece)),
                             '{} +- {} ({})'.format(np.mean(scaled_ece), np.std(scaled_ece), np.min(scaled_ece))])

    for i in range(len(experiments)):

        ece_txt = r'ECE: ${}\pm{}$ %'.format(*[np.round(k, 2) for k in eces[i]])

        _save_path = os.path.join(experiments_path, 'threshold_results', '{}'.
                                  format(experiments[i]['label'].replace('_', ' ')))

        if not os.path.exists(_save_path):
            os.makedirs(_save_path)

        f, axs = plt.subplots(nrows=1, ncols=1)  # , figsize=[4, 3])

        axs.text(0.55, 0.05, ece_txt, transform=axs.transAxes,  bbox=dict(boxstyle='round', facecolor='w', alpha=0.9))

        f.set_figheight(3 * 1)
        f.set_figwidth(4 * 1)

        bars = all_bars[i]
        bars_m = bars.mean(0)
        bars_std = np.std(bars, 0)

        width = 1 / len(bars_m[0])

        x = np.linspace(0 + width / 2, 1 - width / 2, len(bars_m[0]))

        axs.grid(True, linestyle='--', linewidth=1, alpha=0.8)

        axs.plot([0, 1], [0, 1], linestyle='-', label='Perfectly calibrated', c='k', linewidth=1)

        axs.bar(x, bars_m[1], width=width, yerr=bars_std[1], fill=True, linewidth=0, color='b', edgecolor='k',
                error_kw=dict(lw=1, capsize=1, capthick=1), alpha=0.8, zorder=3, label='Outputs')

        axs.bar(x, bars_m[1], width=width, yerr=bars_std[1], fill=False, linewidth=1, color='k', edgecolor='k',
                error_kw=dict(lw=1, capsize=1, capthick=1), alpha=0.8, zorder=3)

        axs.bar(x, x - bars_m[1], width=width, alpha=0.3, fill=True, linewidth=0, color='r', edgecolor='k', zorder=3,
                bottom=bars_m[1], label='Gap')
        axs.bar(x, x - bars_m[1], width=width, fill=False, linewidth=1, edgecolor='darkred', zorder=3, alpha=0.75,
                bottom=bars_m[1])

        axs.set_xlim(0, 1)
        axs.set_ylim(0, 1)
        axs.set_xlabel('confidence')
        axs.set_ylabel('accuracy')

        f.legend(loc='upper left', bbox_to_anchor=(0.13, 0.95), framealpha=1)

        f.savefig(os.path.join(_save_path, "calibration_bars_{}.pdf".format(experiments[i]['network_type'])))
        plt.close(f)

    return experiments_results


if __name__ == '__main__':

    args = sys.argv[1:]

    with open(args[0], "r") as read_file:
        experiments = json.load(read_file)

    results = main(experiments)
