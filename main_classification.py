import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms

import ANN
import DropoutNet
from base import ScaledMixtureGaussian, Gaussian


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

        h, diff,  scores = [], [], []

        for i in r:
            h.append(i[0])
            diff.append(i[1])
            scores.append(i[2])

        h = np.asarray(h).mean(0)
        diff = np.asarray(diff).mean(0)
        scores = np.asarray(scores).mean(0)

        x = range(len(scores))
        ax[0].plot(x, scores, label=d.get('label', d['network_type']), c=d['color'])

        ax[1].plot(x, h[:, 0],# linestyle='--',
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

    experiments_results = []
    adversarial_attack_results = []
    rotation_results = []

    for data in experiments:
        print(data)

        # PRIORS

        prior = None
        if data.get('prior_mu_sigma') is not None:

            mu, sigma = data.get('prior_mu_sigma')
            if data.get('scaled_gaussian', False):
                pi = data['scaled_pi']
                mu1, sigma1 = data['prior1_mu_sigma']
                prior = ScaledMixtureGaussian(pi=pi, mu1=mu, s1=sigma, mu2=mu1, s2=sigma1)
            else:
                prior = Gaussian(mu, sigma)

        # Parameters of the experiments
        batch_size = data.get('batch_size', 64)
        lr = data.get('lr', 1e-3)
        topology = data['topology']
        weights_mu_init = data.get('weights_mu_init', None)
        weights_rho_init = data.get('weights_rho_init', -3)
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
        label = data.get("label", network)
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
                data['color'] = 'red'
            elif network == 'bbb':
                base_model = BBB.BBB
                trainer = BBB.Trainer
                data['color'] = 'green'
            elif network == 'normal':
                base_model = ANN.ANN
                trainer = ANN.Trainer
                data['color'] = 'blue'
            elif network == 'dropout':
                base_model = DropoutNet.Dropnet
                trainer = DropoutNet.Trainer
                data['color'] = 'k'

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

            model.to(device)
            opt = optimizer(model.parameters(), lr=lr)

            current_path = os.path.join(save_path, exp_name)  # , str(e))

            results = {}
            epoch_start = 0

            results_path = os.path.join(current_path, 'results_{}.data'.format(e))

            if save and not os.path.exists(current_path):
                os.makedirs(current_path)

            if load and os.path.exists(results_path):
                checkpoint = torch.load(results_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_start = checkpoint['epoch'] + 1
                results = checkpoint
                # print(results.get('test_results')[1:])
                # print(results.get('train_results'))

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

            if network != 'normal':
                h1 = t.rotation_test(samples=test_samples)
                h2 = t.attack_test(samples=test_samples)

                local_rotate_res.append(h1)
                local_attack_res.append(h2)
            else:
                local_rotate_res.append(None)
                local_attack_res.append(None)

            # if network in ['bbb', 'mmd']:
            #     to_hist = []
            #     last = None
            #     for layer in t.model.features:
            #         if isinstance(layer, (BayesianLinearLayer, BayesianCNNLayer)):
            #             last = layer
            #             w = layer.w
            #             b = layer.b
            #
            #             mean = np.abs(w.mu.detach().cpu().numpy())
            #             std = w.sigma.detach().cpu().numpy()
            #             snr = mean / std
            #             to_hist.append(np.reshape(10 * np.log10(snr), -1))
            #
            #             if b is not None:
            #                 mean = np.abs(b.mu.detach().cpu().numpy())
            #                 std = b.sigma.detach().cpu().numpy()
            #                 snr = mean / std
            #                 to_hist.append(np.reshape(10 * np.log10(snr), -1))
            #
            #     to_hist = np.concatenate(to_hist)
            #     # import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots(nrows=2, ncols=1)
            #     hist, _, _ = ax[0].hist(to_hist, bins=100)
            #     cs = np.cumsum(hist)
            #     ax[1].plot(range(len(cs)), cs)
            #     # plt.show()
            #
            #     plt.savefig(os.path.join(current_path, "{}_{}_weights_distribution.pdf".format(e, network)),
            #                 bbox_inches='tight')
            #     plt.close()
            #
            #     plt.figure()
            #
            #     last_mu = last.w.mu.detach().cpu().numpy().mean(-1)
            #     last_sigma = last.w.sigma.detach().cpu().numpy().mean(-1)
            #
            #     for mu, sigma in zip(last_mu, last_sigma):
            #         # print(mu, sigma)
            #         x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            #         plt.plot(x, stats.norm.pdf(x, mu, sigma))
            #
            #     plt.show()
            #     plt.close()

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
        res = 100 - np.asarray(res)*100

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

