from itertools import chain

import base
# from BMMD import pairwise_distances
from base import ScaledMixtureGaussian, Gaussian
from torchvision import datasets
from torchvision.transforms import transforms
import torch
import matplotlib.pyplot as plt


def get_dataset(name, batch_size, dev_split):
    if name == "fMNIST":
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Normalize((0,), (1,)),
            torch.flatten
        ])

        train_split = datasets.MNIST('./Datasets/MNIST', train=True, download=True,
                                     transform=image_transform)

        if dev_split > 0:
            train_size = int((1 - dev_split) * len(train_split))
            test_size = len(train_split) - train_size

            train_split, dev_split = torch.utils.data.random_split(train_split, [train_size, test_size])

            train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)

            test_loader = torch.utils.data.DataLoader(dev_split, batch_size=batch_size, shuffle=False)

        else:
            train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)

            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./MNIST', train=False, transform=image_transform), batch_size=1000,
                shuffle=False)

        input_size = 784
        classes = 10
        return input_size, classes, train_loader, test_loader


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

        def __str__(self):
            return self.value

    class Datasets(Enum):
        fMNIST = 'fMNIST'

        def __str__(self):
            return self.value

    # print(experiments)
    experiments_results = []

    for data in experiments:
        print(data)
        # PRIORS
        mu, sigma = data['prior_mu_sigma']

        if data['scaled_gaussian']:
            # raise ValueError('Scaled Gaussian not implemented yet')
            pi = data['scaled_pi']
            mu1, sigma1 = data['prior1_mu_sigma']
            # prior = Normal(0, 1)
            prior = ScaledMixtureGaussian(pi=pi, mu1=mu, s1=sigma, mu2=mu1, s2=sigma1)
        else:
            prior = Gaussian(mu, sigma)

        # Parameters of the experiments
        batch_size = data['batch_size']
        lr = data['lr']
        topology = data['topology']
        weights_mu_init = data['weights_mu_init']
        weights_rho_init = data['weights_rho_init']
        optimizer = data['optimizer'].lower()
        dataset = data["dataset"]
        network = data["network_type"].lower()
        experiments = data.get('experiments', 1)
        seeds = data.get('experiments_seeds', 0)
        device = 'cuda' if torch.cuda.is_available() and data['use_cuda'] else 'cpu'
        save_path = data['save_path']
        loss_weights = data['loss_weights']
        epochs = data['epochs']
        train_samples = data['train_samples']
        test_samples = data['test_samples']
        exp_name = data['exp_name']
        save = data['save']
        load = data['load']
        dev_split = data['dev_split']

        if epochs < 0:
            raise ValueError('The number of epoch should be > 0')

        if isinstance(experiments, int):
            experiments = [experiments]

        if isinstance(seeds, int):
            seeds = [seeds]

        if isinstance(experiments, list):
            if (not isinstance(seeds, list)) or (isinstance(seeds, list) and len(experiments) != len(seeds)):
                raise ValueError('The number of the experiments and the number of seeds needs to match, '
                                 'given: {} and {}'.format(experiments, seeds))

        if network not in list(map(str, NetworkTypes)):
            raise ValueError('Supported networks', list(NetworkTypes))
        else:
            if network == 'mmd':
                base_model = BMMD.BMMD
                train_step = BMMD.epoch
            elif network == 'bbb':
                base_model = BBB.BBB
                train_step = BBB.epoch
            elif network == 'normal':
                base_model = base.ANN
                train_step = base.epoch

        if optimizer not in list(map(str, Optimizers)):
            raise ValueError('Supported optimizers', list(Optimizers))
        else:
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD

            elif optimizer == 'adam':
                optimizer = torch.optim.Adam

        run_results = []
        for e, seed in zip(experiments, seeds):

            torch.manual_seed(seed)
            np.random.seed(seed)

            if dataset not in list(map(str, Datasets)):
                raise ValueError('Supported datasets {}, given {}'.format(list(Datasets), dataset))
            else:
                input_size, classes, train_loader, test_loader = get_dataset(dataset, batch_size, dev_split)

            model = base_model(input_size=input_size, prior=prior, mu_init=weights_mu_init, device=device,
                               rho_init=weights_rho_init, topology=topology, classes=classes)
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

            f1 = results.get('test_results', ['not calculated'])[-1]
            f1_train = results.get('train_results', ['not calculated'])[-1]
            progress_bar = tqdm.tqdm(range(epoch_start, epochs), initial=epoch_start, total=epochs)
            progress_bar.set_postfix(f1_test=f1, f1_train=f1_train)

            for i in progress_bar:

                loss, (train_true, train_pred), (test_true, test_pred) = train_step(model=model, optimizer=opt,
                                                                                    train_dataset=train_loader,
                                                                                    test_dataset=test_loader,
                                                                                    train_samples=train_samples,
                                                                                    test_samples=test_samples,
                                                                                    device=device, weights=loss_weights)
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

        print('-' * 200)
        experiments_results.append(run_results)

        # to_hist = []
        # if network in ['mmd', 'bbb']:
        #     for layer in chain(model.features, model.classificator):
        #         w = layer.w
        #         b = layer.b
        #
        #         mean = np.abs(w.mu.detach().cpu().numpy())
        #         std = torch.log(1 + torch.exp(w.rho.detach())).cpu().numpy()
        #         snr = mean/std
        #         to_hist.append(np.reshape(10*np.log10(snr), -1))
        #
        # to_hist = np.concatenate(to_hist)
        #
        # fig, ax = plt.subplots(nrows=2, ncols=1)
        # hist, _, _ = ax[0].hist(to_hist)
        # cs = np.cumsum(hist)
        # ax[1].plot(range(len(cs)), cs)
        # plt.show()

    return experiments_results
