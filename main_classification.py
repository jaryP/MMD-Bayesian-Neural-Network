import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torchvision
import scipy.stats as stats
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from torchvision import datasets
from torchvision.transforms import transforms

import ANN
import DropoutNet
from base import ScaledMixtureGaussian, Gaussian, Laplace
from bayesian_utils import BayesianLinearLayer, BayesianCNNLayer


def get_dataset(name, batch_size, dev_split):
    if name in ['fMNIST', 'MNIST']:
        if name == "fMNIST":
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                # transforms.Normalize((0,), (1,)),
                torch.flatten
            ])
        else:
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                # transforms.Normalize((0,), (1,)),
            ])

        train_split = datasets.MNIST('./Datasets/MNIST', train=True, download=True,
                                     transform=image_transform)
        test_split = datasets.MNIST('./Datasets/MNIST', train=False, download=True,
                                    transform=image_transform)

        sample = train_split[0][0]
        classes = 10

    if name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        train_split = torchvision.datasets.CIFAR10(root='./Datasets/CIFAR10', train=True,
                                                   download=True, transform=transform)

        test_split = torchvision.datasets.CIFAR10(root='./Datasets/CIFAR10', train=False,
                                                  download=True, transform=transform)
        classes = 10
        sample = train_split[0][0]

    if dev_split > 0:
        train_size = int((1 - dev_split) * len(train_split))
        test_size = len(train_split) - train_size

        train_split, dev_split = torch.utils.data.random_split(train_split, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(dev_split, batch_size=batch_size, shuffle=False)

    else:
        train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(test_split, batch_size=batch_size, shuffle=False)

    return sample, classes, train_loader, test_loader


def plot_test(exps, tests):
    fig, ax = plt.subplots(nrows=3)

    for d, r in zip(exps, tests):

        if r[0] is None:
            continue

        h, diff, scores = [], [], []

        for i in r:
            h.append(i[0])
            diff.append(i[1])
            scores.append(i[2])

        h = np.asarray(h).mean(0)
        diff = np.asarray(diff).mean(0)
        scores = np.asarray(scores).mean(0)

        x = range(len(scores))
        ax[0].plot(x, scores, label=d.get('label', d['network_type']), c=d['color'])

        ax[1].plot(x, h[:, 0],  # linestyle='--',
                   label='{} uncertainty'.format(d.get('label', d['network_type'])), c=d['color'])

        # ax[1].plot(x, h[:, 1], linestyle='-.',
        #            label='{} uncertainty'.format(d.get('label', d['network_type'])), c=d['color'])

        ax[2].plot(x, diff,
                   label='{} (top1-top2)**2'.format(d.get('label', d['network_type'])), c=d['color'])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')

    return fig, ax


def main(experiments):
    import sklearn.metrics as metrics
    import tqdm as tqdm

    import BBB
    import BMMD

    import os
    from enum import Enum
    import numpy as np

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    class Optimizers(Enum):
        sgd = 'sgd'
        adam = 'adam'

        def __str__(self):
            return self.value

    class NetworkTypes(Enum):
        bbb = 'bbb'
        mmd = 'mmd'
        normal = 'normal'
        dropout = 'dropout'

        def __str__(self):
            return self.value

    class Datasets(Enum):
        fMNIST = 'fMNIST'
        MNIST = 'MNIST'
        CIFAR10 = 'CIFAR10'

        def __str__(self):
            return self.value

    class Priors(Enum):
        gaussian = 'gaussian'
        scaledGaussian = 'scaledGaussian'
        laplace = 'laplace'
        # multivaraite = 'Mu'

        def __str__(self):
            return self.value

    experiments_results = []
    adversarial_attack_results = []
    rotation_results = []
    hists = []
    all_ws = []

    for data in experiments:
        print(data)

        # PRIORS

        prior = None
        data_prior = data.get('prior')

        if data_prior:
            t = data_prior.get('type')
            if t not in list(map(str, Priors)):
                raise ValueError('Supported priors', list(Priors))
            else:
                if t == 'gaussian':
                    prior = Gaussian(data_prior['mu'], data_prior['sigma'])
                if t == 'laplace':
                    prior = Laplace(data_prior['mu'], data_prior['scale'])
                if t == 'scaledGaussian':
                    prior = ScaledMixtureGaussian(pi=data_prior['phi'], mu1=data_prior['mu1'], s1=data_prior['sigma1'],
                                                  mu2=data_prior['mu2'], s2=data_prior['sigma2'])

        # if data.get('prior_mu_sigma') is not None:
        #
        #     mu, sigma = data.get('prior_mu_sigma')
        #     if data.get('scaled_gaussian', False):
        #         pi = data['scaled_pi']
        #         mu1, sigma1 = data['prior1_mu_sigma']
        #         prior = ScaledMixtureGaussian(pi=pi, mu1=mu, s1=sigma, mu2=mu1, s2=sigma1)
        #         # print(prior.s1, prior.s2)
        #         # a = prior.sample(torch.tensor([1000])).numpy()
        #         # plt.hist(a, bins=100)
        #         #
        #         # # prior = Gaussian(mu, sigma)
        #         # # a = prior.sample(torch.tensor([1000])).numpy()
        #         # # plt.hist(a, bins=100)
        #         # #
        #         # # prior = Gaussian(mu1, sigma1)
        #         # # a = prior.sample(torch.tensor([1000])).numpy()
        #         # # plt.hist(a, bins=100)
        #         #fprin
        #         # plt.show()
        #
        #     else:
        #         prior = Gaussian(mu, sigma)

        # Parameters of the experiments
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
        loss_weights = data.get('loss_weights', {})
        epochs = data['epochs']
        train_samples = data.get('train_samples', 2)
        test_samples = data.get('test_samples', 2)
        exp_name = data['exp_name']
        save = data['save']
        load = data['load']
        dev_split = data.get('dev_split', 0)
        local_trick = data.get('local_trick', False)
        # label = data.get("label", network)
        network_parameters = data.get('network_parameters', {})

        if epochs < 0:
            raise ValueError('The number of epoch should be > 0')

        # if isinstance(experiments, int):
        #     experiments = [experiments]

        if isinstance(seeds, int):
            seeds = [seeds]

        # if isinstance(experiments, list):
        #     if (not isinstance(seeds, list)) or (isinstance(seeds, list) and len(experiments) != len(seeds)):
        #         raise ValueError('The number of the experiments and the number of seeds needs to match, '
        #                          'given: {} and {}'.format(experiments, seeds))

        if network not in list(map(str, NetworkTypes)):
            raise ValueError('Supported networks', list(NetworkTypes))
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
            data["color"] = None

        if optimizer not in list(map(str, Optimizers)):
            raise ValueError('Supported optimizers', list(Optimizers))
        else:
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD

            elif optimizer == 'adam':
                optimizer = torch.optim.Adam

        run_results = []
        local_rotate_res = []
        local_attack_res = []

        for e, seed in enumerate(seeds):

            torch.manual_seed(seed)
            np.random.seed(seed)

            if dataset not in list(map(str, Datasets)):
                raise ValueError('Supported datasets {}, given {}'.format(list(Datasets), dataset))
            else:
                sample, classes, train_loader, test_loader = get_dataset(dataset, batch_size, dev_split)

            model = base_model(prior=prior, mu_init=weights_mu_init, device=device,
                               rho_init=weights_rho_init, topology=topology, classes=classes, local_trick=local_trick,
                               sample=sample, **network_parameters)
            init_model = deepcopy(model)

            model.to(device)
            opt = optimizer(model.parameters(), lr=lr)

            current_path = os.path.join(save_path, exp_name)  # , str(e))

            results = {}
            epoch_start = 0

            results_path = os.path.join(current_path, 'results_{}.data'.format(e))

            if save and not os.path.exists(current_path):
                os.makedirs(current_path)

            loaded = False

            if load and os.path.exists(results_path):
                loaded = True
                checkpoint = torch.load(results_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_start = checkpoint['epoch'] + 1
                results = checkpoint
                print(results.get('test_results')[1:])
                print(results.get('train_results'))

            f1 = results.get('test_results', ['not calculated'])[-1]
            f1_train = results.get('train_results', ['not calculated'])[-1]
            progress_bar = tqdm.tqdm(range(epoch_start, epochs), initial=epoch_start, total=epochs)
            progress_bar.set_postfix(f1_test=f1, f1_train=f1_train)

            t = trainer(model, train_loader, test_loader, opt)

            for i in progress_bar:

                if i == 0:
                    (test_true, test_pred) = t.test_evaluation(samples=test_samples)

                    f1 = metrics.f1_score(test_true, test_pred, average='micro')

                    epochs_res = results.get('test_results', [])
                    epochs_res.append(f1)

                    epochs_res_train = results.get('train_results', [])
                    epochs_res_train.append(f1_train)

                    results.update({'epoch': i, 'test_results': epochs_res})

                loss, (train_true, train_pred), (test_true, test_pred) = t.train_step(train_samples=train_samples,
                                                                                      test_samples=test_samples,
                                                                                      weights=loss_weights)

                loss = np.mean(loss)

                f1 = metrics.f1_score(test_true, test_pred, average='micro')

                f1_train = metrics.f1_score(train_true, train_pred, average='micro')

                progress_bar.set_postfix(f1_test=f1, f1_train=f1_train)

                epochs_res = results.get('test_results', [])
                epochs_res.append(f1)

                epochs_res_train = results.get('train_results', [])
                epochs_res_train.append(f1_train)

                losses = results.get('losses', [])
                losses.append(loss)

                results.update({
                    'epoch': i, 'test_results': epochs_res, 'train_results': epochs_res_train,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses
                })
                if save:
                    torch.save(results, results_path)

            run_results.append(results)
            progress_bar.close()

            # plt.figure(0)

            h1 = None
            h2 = None

            if network != 'normal':

                attack1_path = os.path.join(current_path, 'attack1_{}.data'.format(e))
                attack2_path = os.path.join(current_path, 'attack2_{}.data'.format(e))

                if loaded:
                    if os.path.exists(attack1_path):
                        with open(attack1_path, "rb") as f:
                            h1 = pickle.load(f)

                    if os.path.exists(attack2_path):
                        with open(attack2_path, "rb") as f:
                            h2 = pickle.load(f)

                if h1 is None:
                    h1 = t.rotation_test(samples=test_samples)
                    if save:
                        with open(attack1_path, "wb") as output_file:
                            pickle.dump(h1, output_file)

                # if h2 is None:
                #     h2 = t.attack_test(samples=test_samples)
                #     if save:
                #         with open(attack2_path, "wb") as output_file:
                #             pickle.dump(h2, output_file)

            local_rotate_res.append(h1)
            local_attack_res.append(h2)

            # else:
            #     local_rotate_res.append(None)
            #     local_attack_res.append(None)

            ws = []
            ws_init = []
            to_hist = []

            if network in ['bbb', 'mmd']:
                # last = None
                for layer in t.model.features:
                    if isinstance(layer, (BayesianLinearLayer, BayesianCNNLayer)):
                        last = layer
                        w = layer.w
                        b = layer.b

                        # snr = w.weights.detach().cpu().numpy()
                        mean = np.abs(w.mu.detach().cpu().numpy())
                        std = w.sigma.detach().cpu().numpy()
                        # mean = np.abs(snr.mean())
                        # std = np.std(snr)
                        snr = mean / std
                        to_hist.append(np.reshape(snr, -1))
                        ws.append(np.reshape(w.weights.detach().cpu().numpy(), -1))

                        if b is not None:
                            mean = np.abs(b.mu.detach().cpu().numpy())
                            std = b.sigma.detach().cpu().numpy()
                            snr = mean / std
                            to_hist.append(np.reshape(snr, -1))
                            ws.append(np.reshape(b.weights.detach().cpu().numpy(), -1))

                for layer in init_model.features:
                    if isinstance(layer, (BayesianLinearLayer, BayesianCNNLayer)):
                        # last = layer
                        w = layer.w
                        b = layer.b

                        # snr = w.weights.detach().cpu().numpy()
                        # mean = np.abs(w.mu.detach().cpu().numpy())
                        # std = w.sigma.detach().cpu().numpy()
                        # mean = np.abs(snr.mean())
                        # std = np.std(snr)
                        # snr = mean / std
                        # to_hist.append(np.reshape(snr, -1))
                        ws_init.append(np.reshape(w.weights.detach().cpu().numpy(), -1))

                        if b is not None:
                            # mean = np.abs(b.mu.detach().cpu().numpy())
                            # std = b.sigma.detach().cpu().numpy()
                            # snr = mean / std
                            # to_hist.append(np.reshape(snr, -1))
                            ws_init.append(np.reshape(b.weights.detach().cpu().numpy(), -1))

                ws = np.concatenate(ws)
                ws_init = np.concatenate(ws_init)

                to_hist = np.concatenate(to_hist)
                hists.append(to_hist)
                all_ws.append(ws)

                # print(last.w.weights.shape)
                # c0 = []
                # c1 = []
                # plt.figure()
                #
                # # for i in range(100):
                # w = last.w.weights.detach().cpu().numpy()
                # # print(last.w.sigma.detach().cpu().numpy()[:, 0])
                #
                # # print(last.w.mu.detach().cpu().numpy()[:, 0])
                #
                # plt.hist2d(w[:, 0], w[:, 1], bins=10000)
                # plt.show()

                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(nrows=2, ncols=1)
                # hist, _, _ = ax[0].hist(to_hist, bins=100)
                # cs = np.cumsum(hist)
                # ax[1].plot(range(len(cs)), cs)
                # # plt.show()
                #
                # plt.savefig(os.path.join(current_path, "{}_{}_weights_distribution.pdf".format(e, network)),
                #             bbox_inches='tight')
                # plt.close()

                # plt.figure()
                #
                # last_mu = last.w.mu.detach().cpu().numpy().mean(-1)
                # last_sigma = last.w.sigma.detach().cpu().numpy().mean(-1)
                #
                # for mu, sigma in zip(last_mu, last_sigma):
                #     # print(mu, sigma)
                #     x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
                #     plt.plot(x, stats.norm.pdf(x, mu, sigma))
                #
                # plt.show()
                # plt.close()
            else:
                hists.append([])
                for m in t.model.parameters():
                    ws.append(np.reshape(m.data.detach().cpu().numpy(), -1))

                for m in init_model.parameters():
                    ws_init.append(np.reshape(m.data.detach().cpu().numpy(), -1))

                ws = np.concatenate(ws)
                ws_init = np.concatenate(ws_init)

            fig, ax = plt.subplots()
            hist, _, _ = ax.hist(ws, bins=1000, linewidth=0.5, density=False,
                                 label="Final weights",
                                 color=data['color'], histtype='step')

            hist, _, _ = ax.hist(ws_init, bins=1000, linewidth=0.5, density=False,
                                 label='Initial weights', histtype='step')

            if network in ['bbb', 'mmd']:
                hist, _, _ = ax.hist(prior.sample(torch.Size([len(ws)])).numpy(), bins=1000, linewidth=0.5, density=False,
                                     label="Prior", histtype='step')

            plt.legend()
            fig.savefig(os.path.join(current_path, "{}_{}_posterior.pdf".format(e, network)),
                        bbox_inches='tight')
            plt.close(fig)

        # print(np.asarray(local_rotate_res))
        # print(np.asarray(local_rotate_res).mean(0))

        # print(np.asarray(local_rotate_res).shape)
        rotation_results.append(local_rotate_res)
        adversarial_attack_results.append(local_attack_res)
        # print(rotation_results[-1].shape)
        # adversarial_attack_results.append(np.asarray(local_attack_res).mean(0))

        print('-' * 200)
        experiments_results.append(run_results)

        # f1 = results['test_results']
        # plt.plot(range(len(f1)), f1, label=label)
        # plt.legend()
        # print(f1)

    # plt.show()

    fig, ax = plot_test(experiments, rotation_results)
    for a in ax:
        a.set_xticklabels(['']+ANN.Trainer.rotations)
    # fig.draw()
    fig.savefig(os.path.join(save_path, "rotation.pdf"), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plot_test(experiments, adversarial_attack_results)
    for a in ax:
        a.set_xticklabels(['']+ANN.Trainer.epsilons)
    fig.savefig(os.path.join(save_path, "fgsa.pdf"), bbox_inches='tight')
    plt.close(fig)

    plt.figure(0)
    for d, r in zip(experiments, experiments_results):
        res = [i['test_results'][1:] for i in r]
        res = 100 - np.asarray(res) * 100
        # print(res.shape, res.mean(0), res.std(0))
        means = res.mean(0)
        stds = res.std(0)

        f1 = means

        plt.plot(range(len(f1)), f1, label=d.get('label', d['network_type']), c=d['color'])

        plt.fill_between(range(len(f1)), f1 - stds, f1 + stds, alpha=0.1, color=d['color'])

        plt.legend()

    plt.xticks(range(len(f1)), [i + 1 for i in range(len(f1))])
    # print(f1)
    # plt.ylim(0.95, 1)

    plt.ylabel("Error test (%)")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(save_path, "score.pdf"), bbox_inches='tight')
    plt.close(0)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    # ws_max = np.asarray(ws).max()
    for d, r in zip(experiments, hists):
        if len(r) == 0:
            continue

        # hist, _, _ = ax.hist(10 * np.log10(r + 1e-12), bins=1000, linewidth=0.5,
        #                      label=d.get('label', d['network_type']), color=d['color'], histtype='step')
        r = 10 * np.log10(r + 1e-12)
        xs = np.linspace(-60, 60, 200)
        density = gaussian_kde(r)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        plt.plot(xs, density(xs), label=d.get('label', d['network_type']), color=d['color'])
        # cs = np.cumsum(hist)
        # ax[1].plot(range(len(cs)), cs, color=d['color'], linewidth=0.5)
        # if '3' in d['label']:
        # hist, _, _ = ax[1].hist(ws, bins=2000, linewidth=0.5, density=True,
        #                         label=d.get('label', d['network_type']), color=d['color'], histtype='step')

    ax.legend(loc='upper left')
    # ax[1].legend(loc='upper left')
    # cs = np.cumsum(hist)
    # ax[1].plot(range(len(cs)), cs)
    fig.savefig(os.path.join(save_path, "hists.pdf"), bbox_inches='tight')
    plt.close(fig)

    return experiments_results


if __name__ == '__main__':
    import json
    import sys
    import numpy as np

    args = sys.argv[1:]

    with open(args[0], "r") as read_file:
        experiments = json.load(read_file)

    # print(len(experiments))
    results = main(experiments)

    # fig = plt.figure()
    # for d, r in zip(experiments, results):
    #     res = [i['test_results'][1:] for i in r]
    #     res = np.asarray(res)
    #
    #     # print(res.shape, res.mean(0), res.std(0))
    #     means = res.mean(0)
    #     stds = res.std(0)
    #
    #     f1 = means
    #
    #     plt.plot(range(len(f1)), f1, label=d.get('label'))
    #
    #     plt.fill_between(range(len(f1)), f1 - stds, f1 + stds, alpha=0.3)
    #
    #     plt.legend()
    #     # print(f1)
    # # plt.ylim(0.95, 1)
    #
    # plt.ylabel("Error test (%)")
    # plt.xlabel("Epochs")
    # plt.show()
