import GPy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import json
import sys
import numpy as np

import ANN
import DropoutNet
from priors import Gaussian, Laplace, ScaledMixtureGaussian, Uniform

font = {'font.family': 'serif',
        'axes.labelsize': 11,
        'font.size': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': [4, 3],
        'text.usetex': True,
        'font.serif': ['Times New Roman'] + plt.rcParams['font.serif']}

plt.rcParams.update(font)


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
    y = (y - y.mean())

    test_idx = np.concatenate((np.arange(0, regression_points // 2),
                           np.arange(regression_points // 2 + regression_points, regression_points * 2)))

    train_idx = np.arange(regression_points // 2, regression_points // 2 + regression_points)

    train_dataset = RegressionDataset(x[train_idx],
                                      y[train_idx])

    test_dataset = RegressionDataset(x[test_idx],
                                     y[test_idx])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset[0][0]


def main(experiment):
    import tqdm as tqdm

    import BBB
    import BMMD

    import os
    import numpy as np

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Optimizers = ['sgd', 'adam', 'rmsprop']
    NetworkTypes = ['bbb', 'mmd', 'normal', 'dropout']
    Datasets = ['homo', 'hetero']
    # PosteriorType = ['layers', 'neurons', 'weights', 'multiplicative']
    # LrScheduler = ['step', 'exponential', 'plateau']
    Priors = ['gaussian', 'laplace', 'uniform', 'scaledGaussian']

    experiments_results = []

    # experiments_path = experiment['experiments_path']

    # experiments = experiment['experiments']

    # print(cm.colors)
    for data in experiment:
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

        regression_points = data.get('regression_points', 100)

        variance = data['variance']
        noise = data['noise']
        points_range = data.get('range', [-10, 10])
        batch_size = data.get('batch_size', 10)

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

        if isinstance(seeds, int):
            seeds = [seeds]

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

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for e, seed in enumerate(seeds):

            seed = seed + int(noise*100)
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

            results = {}
            epoch_start = 0

            progress_bar = tqdm.tqdm(range(epoch_start, epochs), initial=epoch_start, total=epochs)
            progress_bar.set_postfix(loss='')

            t = trainer(model, train_loader, test_loader, opt)

            for _ in progress_bar:
                loss, _, _ = t.train_step(train_samples=train_samples, test_samples=test_samples,
                                          weights=loss_weights)
                progress_bar.set_postfix(loss=np.mean(loss))

            with torch.no_grad():
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

            mx = -np.inf
            mn = +np.inf

            for x, y in train_loader:
                mn = min(y.min().item(), mn)
                mx = max(y.max().item(), mx)
                plt.scatter(x, y, c='b', marker='+', alpha=0.5, s=1)

            for x, y in test_loader:
                mn = min(y.min().item(), mn)
                mx = max(y.max().item(), mx)
                plt.scatter(x, y, c='r', s=4)

            plt.fill_between(x_true, y_pred - y_noise, y_pred + y_noise, alpha=0.2)
            plt.fill_between(x_true, y_pred - 2 * y_noise, y_pred + 2 * y_noise, alpha=0.2)
            plt.fill_between(x_true, y_pred - 3 * y_noise, y_pred + 3 * y_noise, alpha=0.2)

            plt.plot(x_true, y_pred, linewidth=0.8)
            offset = np.abs(mn - mx) * 0.1

            plt.ylim(mn - offset, mx + offset)
            plt.xlim(np.min(x_true), np.max(x_true))

            plt.grid(True, alpha=0.2)


            plt.xlabel('x')
            plt.ylabel('y')

            plt.savefig(os.path.join(save_path, "{}_{}.pdf".format(e, network)), bbox_inches='tight')
            plt.close()

            run_results.append(results)
            progress_bar.close()

        print('-' * 200)
        experiments_results.append(run_results)

    return experiments_results


if __name__ == '__main__':
    args = sys.argv[1:]

    with open(args[0], "r") as read_file:
        experiments = json.load(read_file)

    results = main(experiments)
