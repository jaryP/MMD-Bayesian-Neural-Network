import GPy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

import ANN
import DropoutNet
from priors import Gaussian, Laplace, ScaledMixtureGaussian, Uniform


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

    # x = np.random.uniform(points_range[0], points_range[1], regression_points)[:, None]
    dist = np.abs(points_range[0] - points_range[1]) // 2

    x = np.concatenate((
        np.random.uniform(points_range[0] - dist, points_range[0], regression_points // 2)[:, None],
        np.random.uniform(points_range[0], points_range[1], regression_points)[:, None],
        np.random.uniform(points_range[1], points_range[1] + dist, regression_points // 2)[:, None]
    ))

    x.sort(axis=0)

    k = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    C = k.K(x, x) + np.eye(x.shape[0]) * (x) ** 2 * noise ** 2

    y = np.random.multivariate_normal(np.zeros(x.shape[0]), C)[:, None]
    y = (y - y.mean())  # + np.random.normal(scale=noise, size=y.shape)

    test_idx = np.concatenate((np.arange(0, regression_points // 2),
                           np.arange(regression_points // 2 + regression_points, regression_points * 2)))

    train_idx = np.arange(regression_points // 2, regression_points // 2 + regression_points)

    train_dataset = RegressionDataset(x[train_idx],
                                      y[train_idx])

    test_dataset = RegressionDataset(x[test_idx],
                                     y[test_idx])
    # lengthscale = 1
    # # variance = 2
    # # sig_noise = 0.1
    #
    # x = np.random.uniform(points_range[0]-2, points_range[0], regression_points//2)[:, None]
    # x = np.concatenate((x, np.random.uniform(points_range[1], points_range[1]+2, regression_points//2)[:, None]))
    # x.sort(axis=0)
    #
    # k = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    # C = k.K(x, x) + np.eye(regression_points) * (x + 2) ** 2 * noise ** 2
    #
    # y = np.random.multivariate_normal(np.zeros(regression_points), C)[:, None]
    # y = (y - y.mean())
    # test_dataset = RegressionDataset(x, y)

    # idx = np.arange(regression_points)
    #
    # np.random.shuffle(idx)
    #
    # split = int(len(idx) * 0.75)
    #
    # x_tr, x_te = x[idx[:split]], x[idx[split:]]
    # y_tr, y_te = y[idx[:split]], y[idx[split:]]
    #
    # train_dataset = RegressionDataset(x_tr, y_tr)
    # test_dataset = RegressionDataset(x_te, y_te)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset[0][0]


def main(experiment):
    import tqdm as tqdm

    import BBB
    import BMMD

    import os
    from enum import Enum
    import numpy as np

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Optimizers = ['sgd', 'adam', 'rmsprop']
    NetworkTypes = ['bbb', 'mmd', 'normal', 'dropout']
    Datasets = ['homo', 'hetero']
    PosteriorType = ['layers', 'neurons', 'weights', 'multiplicative']
    # LrScheduler = ['step', 'exponential', 'plateau']
    Priors = ['gaussian', 'laplace', 'uniform', 'scaledGaussian']

    experiments_results = []

    # experiments_path = experiment['experiments_path']

    # experiments = experiment['experiments']

    # print(cm.colors)
    for data in experiments:
        print(data['save_path'], data['exp_name'])

        if data.get('skip', False):
            continue

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

        regression_points = data.get('regression_points', 100)

        variance = data['variance']
        noise = data['noise']
        points_range = data.get('range', [-10, 10])
        batch_size = data.get('batch_size', regression_points//5)

        # PRIORS

        prior = None
        data_prior = data.get('prior')

        if data_prior:
            t = data_prior.get('type')
            if t not in Priors:
                raise ValueError('Supported priors', list(Priors))
            else:
                if t == 'gaussian':
                    prior = Gaussian(data_prior['mu'], data_prior['sigma'])
                elif t == 'laplace':
                    prior = Laplace(data_prior['mu'], data_prior['scale'])
                elif t == 'scaledGaussian':
                    prior = ScaledMixtureGaussian(pi=data_prior['phi'], mu1=data_prior['mu1'], s1=data_prior['sigma1'],
                                                  mu2=data_prior['mu2'], s2=data_prior['sigma2'])
                elif t == 'uniform':
                    a, b = data_prior['a'], data_prior['b']
                    if network == 'bbb':
                        a = torch.tensor([float(a)])
                        b = torch.tensor([float(b)])
                    prior = Uniform(a=a, b=b)

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

        if network not in NetworkTypes:
            raise ValueError('Supported networks', NetworkTypes)
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

        if optimizer not in Optimizers:
            raise ValueError('Supported optimizers', Optimizers)
        else:
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD

            elif optimizer == 'adam':
                optimizer = torch.optim.Adam

            elif optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop

        run_results = []
        local_rotate_res = []
        local_ettack_res = []

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for e, seed in enumerate(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            if dataset not in Datasets:
                raise ValueError('Supported datasets {}, given {}'.format(Datasets, dataset))
            else:
                train_loader, test_loader, sample = get_dataset(batch_size, regression_points=regression_points,
                                                                variance=variance, noise=noise,
                                                                points_range=points_range)
                if dataset == 'homo':
                    classes = 1
                else:
                    classes = 2

            model = base_model(prior=prior, mu_init=weights_mu_init, device=device, regression=True,
                               rho_init=weights_rho_init, topology=topology, classes=classes, local_trick=local_trick,
                               sample=sample, **network_parameters, posterior_type=posterior_type)

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
                # dist = int(np.abs(points_range[0] - points_range[1]) / 4)
                dist = np.abs(points_range[0] - points_range[1]) // 2
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
                    y_pred, y_noise = pred[:, 0], np.asarray([t.model.noise.exp().item()] * len(x_true))
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
                plt.scatter(x, y, c='b', marker='+', alpha=0.5, s=2)

            for x, y in test_loader:
                mn = min(y.min().item(), mn)
                mx = max(y.max().item(), mx)
                plt.scatter(x, y, c='r', marker='.')

            print(mn, mx)
            # plt.show()
            # print(n)

            plt.fill_between(x_true, y_pred - y_noise, y_pred + y_noise, alpha=0.2)
            plt.fill_between(x_true, y_pred - 2 * y_noise, y_pred + 2 * y_noise, alpha=0.2)
            plt.fill_between(x_true, y_pred - 3 * y_noise, y_pred + 3 * y_noise, alpha=0.2)

            plt.plot(x_true, y_pred)
            offset = np.abs(mn - mx) * 0.1
            plt.ylim(mn - offset, mx + offset)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=17)
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
