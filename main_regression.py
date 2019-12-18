import GPy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

import ANN
import DropoutNet
from base import ScaledMixtureGaussian, Gaussian


class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img, target = self.x[index], self.y[index]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def get_dataset(batch_size, regression_points, points_range, variance, noise):
    lengthscale = 1
    # variance = 2
    # sig_noise = 0.1

    x = np.random.uniform(points_range[0], points_range[1], regression_points)[:, None]
    x.sort(axis=0)

    k = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    C = k.K(x, x) + np.eye(regression_points) * (x + 2) ** 2 * noise ** 2

    y = np.random.multivariate_normal(np.zeros(regression_points), C)[:, None]
    y = (y - y.mean())
    idx = np.arange(regression_points)

    np.random.shuffle(idx)

    split = int(len(idx) * 0.8)

    x_tr, x_te = x[idx[:split]], x[idx[split:]]
    y_tr, y_te = y[idx[:split]], y[idx[split:]]

    train_dataset = RegressionDataset(x_tr, y_tr)
    test_dataset = RegressionDataset(x_te, y_te)

    # if name in ['fMNIST', 'MNIST']:
    #     if name == "fMNIST":
    #         image_transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.1307,), (0.3081,)),
    #             # transforms.Normalize((0,), (1,)),
    #             torch.flatten
    #         ])
    #     else:
    #         image_transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.1307,), (0.3081,)),
    #             # transforms.Normalize((0,), (1,)),
    #         ])
    #
    #     train_split = datasets.MNIST('./Datasets/MNIST', train=True, download=True,
    #                                  transform=image_transform)
    #     test_split = datasets.MNIST('./Datasets/MNIST', train=False, download=True,
    #                                 transform=image_transform)
    #
    #     sample = train_split[0][0]
    #     classes = 10
    #
    # if name == 'CIFAR10':
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    #
    #     train_split = torchvision.datasets.CIFAR10(root='./Datasets/CIFAR10', train=True,
    #                                                download=True, transform=transform)
    #
    #     test_split = torchvision.datasets.CIFAR10(root='./Datasets/CIFAR10', train=False,
    #                                               download=True, transform=transform)
    #     classes = 10
    #     sample = train_split[0][0]
    #
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main(experiments):
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
        hetero = 'hetero'
        homo = 'homo'

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
        local_trick = data.get('local_trick', False)
        label = data.get("label", network)
        network_parameters = data.get('network_parameters', {})
        regression_points = data.get('regression_points', 500)
        batch_size = regression_points
        noise = data.get('noise', 0.1)
        points_range = data.get('range', [-10, 10])
        variance = data.get('variance', 1)

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
        local_ettack_res = []

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for e, seed in enumerate(seeds):

            torch.manual_seed(seed)
            np.random.seed(seed)

            if dataset not in list(map(str, Datasets)):
                raise ValueError('Supported datasets {}, given {}'.format(list(Datasets), dataset))
            else:
                train_loader, test_loader = get_dataset(batch_size, regression_points=regression_points,
                                                        variance=variance, noise=noise, points_range=points_range)
                if dataset == 'homo':
                    classes = 1
                else:
                    classes = 2

            model = base_model(prior=prior, mu_init=weights_mu_init, device=device, regression=True,
                               rho_init=weights_rho_init, topology=topology, classes=classes, local_trick=local_trick,
                               sample=torch.tensor([0.0], dtype=torch.float32), **network_parameters)

            model.to(device)
            # print([(n, p.device) for n, p in model.named_parameters()])
            opt = optimizer(model.parameters(), lr=lr)

            current_path = os.path.join(save_path, exp_name)  # , str(e))

            results = {}
            epoch_start = 0


            progress_bar = tqdm.tqdm(range(epoch_start, epochs), initial=epoch_start, total=epochs)
            progress_bar.set_postfix(loss='')

            t = trainer(model, train_loader, test_loader, opt)

            for i in progress_bar:

                loss, _, _ = t.train_step(train_samples=train_samples, test_samples=test_samples,
                                          weights=loss_weights)
                progress_bar.set_postfix(loss=np.mean(loss))

            with torch.no_grad():
                dist = int(np.abs(points_range[0] - points_range[1]) / 4)
                x_true = torch.linspace(points_range[0] - dist, points_range[1] + dist, 500)
                pred = t.model.eval_forward(x_true[:, None].to(device), samples=test_samples)
                x_true = x_true.cpu().numpy()

                pred = pred.cpu().numpy()

                if len(pred.shape) > 2:
                    pred = pred.mean(0)

                if classes == 2:
                    pred[:, 1:] = np.exp(pred[:, 1:])
                    y_pred, y_noise = pred[:, 0], pred[:, 1]
                else:
                    y_pred, y_noise = pred[:, 0], np.asarray([t.model.noise.exp().item()]*len(x_true))
            # print(pred)

            # fig = plt.figure()

            # for x, y in train_loader:
            #     plt.scatter(x, y, c='r')

            # x_true, y_true, y_pred, n = t.test_evaluation()
            #
            # if isinstance(n, float):
            #     n = [[n]*len(x_true)]
            #
            # x_true = np.reshape(np.asarray(x_true), -1)#.mean(-1)
            # sort_index = np.argsort(x_true)
            #
            # y_true = np.reshape(np.asarray(y_true), -1)#.mean(-1)
            # y_pred = np.reshape(np.asarray(y_pred), -1)#.mean(-1)
            # n = np.reshape(np.asarray(n), -1)**0.5 #.mean(-1)
            #
            # # n = (n ** 2).mean(axis=0) ** 0.5
            #
            # x_true = x_true[sort_index]
            # y_true = y_true[sort_index]
            # y_pred = y_pred[sort_index]

            mx = -np.inf
            mn = +np.inf
            for x, y in train_loader:
                mn = min(y.min().item(), mn)
                mx = max(y.max().item(), mx)
                plt.scatter(x, y, c='b')

            for x, y in test_loader:
                mn = min(y.min().item(), mn)
                mx = max(y.max().item(), mx)
                plt.scatter(x, y, c='r')

            print(mn, mx)
            # plt.show()
            # print(n)

            plt.fill_between(x_true, y_pred - y_noise, y_pred + y_noise, alpha=0.2)
            plt.fill_between(x_true, y_pred - 2*y_noise, y_pred + 2*y_noise, alpha=0.2)
            plt.fill_between(x_true, y_pred - 3*y_noise, y_pred + 3*y_noise, alpha=0.2)

            plt.plot(x_true, y_pred)
            plt.ylim(mn-1, mx+1)
            plt.savefig(os.path.join(save_path, "{}_{}.pdf".format(e, network)), bbox_inches='tight')
            plt.close()

            run_results.append(results)
            progress_bar.close()

        print('-' * 200)
        experiments_results.append(run_results)

        # f1 = results['test_results']
        # plt.plot(range(len(f1)), f1, label=label)
        # plt.legend()
        # print(f1)

    # plt.show()

    # fig, ax = plot_test(experiments, rotation_results)
    # for a in ax:
    #     a.set_xticklabels(['']+ANN.Trainer.rotations)
    # # fig.draw()
    # fig.savefig(os.path.join(save_path, "rotation.pdf"), bbox_inches='tight')
    # plt.close(fig)
    #
    # fig, ax = plot_test(experiments, adversarial_attack_results)
    # for a in ax:
    #     a.set_xticklabels(['']+ANN.Trainer.epsilons)
    # fig.savefig(os.path.join(save_path, "fgsa.pdf"), bbox_inches='tight')
    # plt.close(fig)

    # plt.figure(0)
    # for d, r in zip(experiments, experiments_results):
    #     res = [i['test_results'][1:] for i in r]
    #     res = 100 - np.asarray(res)*100
    #
    #     # print(res.shape, res.mean(0), res.std(0))
    #     means = res.mean(0)
    #     stds = res.std(0)
    #
    #     f1 = means
    #
    #     plt.plot(range(len(f1)), f1, label=d.get('label', d['network_type']), c=d['color'])
    #
    #     plt.fill_between(range(len(f1)), f1 - stds, f1 + stds, alpha=0.1, color=d['color'])
    #
    #     plt.legend()
    #
    #     # print(f1)
    # # plt.ylim(0.95, 1)
    #
    # plt.ylabel("Error test (%)")
    # plt.xlabel("Epochs")
    # plt.savefig(os.path.join(save_path, "score.pdf"), bbox_inches='tight')

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
