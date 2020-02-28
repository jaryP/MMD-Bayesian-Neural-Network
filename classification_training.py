import pickle
import time
import json
import sys
import numpy as np

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import transforms
import random
from shutil import copy
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

    experiments_results = []

    experiments_path = experiment['experiments_path']

    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)

    experiments = experiment['experiments']

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
        network_parameters = data.get('network_parameters', {})
        posterior_type = data.get('posterior_type', 'weights')

        moment = data.get('moment', 0)
        early_stopping_tolerance = data.get('early_stopping_tolerance', 3)
        resize = data.get('resize', None)

        weight_decay = data.get('weight_decay', 1e-5 if network == 'dropout' else 0)

        lr_scheduler = data.get('lr_scheduler', None)

        train_subset = data.get('train_subset', 0)
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

        if train_subset < 0 or train_subset > 1:
            raise ValueError('The train subset to sample needs be a percentage (> 0 and < 1)')

        if isinstance(seeds, int):
            seeds = [seeds]

        if posterior_type not in PosteriorType:
            raise ValueError('Supported posterior_type', PosteriorType)

        if network not in NetworkTypes:
            raise ValueError('Supported networks', NetworkTypes)
        else:
            if network == 'mmd':
                base_model = BMMD.BMMD
                trainer = BMMD.Trainer

            elif network == 'bbb':
                base_model = BBB.BBB
                trainer = BBB.Trainer

            elif network == 'normal':
                base_model = ANN.ANN
                trainer = ANN.Trainer

            elif network == 'dropout':
                base_model = DropoutNet.Dropnet
                trainer = DropoutNet.Trainer

        if optimizer not in Optimizers:
            raise ValueError('Supported optimizers', Optimizers)
        else:
            if optimizer == 'sgd':
                _opt = torch.optim.SGD

            elif optimizer == 'adam':
                _opt = torch.optim.Adam

            elif optimizer == 'rmsprop':
                _opt = torch.optim.RMSprop

        run_results = []

        for e, seed in tqdm.tqdm(enumerate(seeds), total=len(seeds)):

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            best_score = 0

            if dataset not in Datasets:
                raise ValueError('Supported datasets {}, given {}'.format(Datasets, dataset))
            else:
                sample, classes, train_loader, test_loader = get_dataset(dataset, batch_size,
                                                                         dev_split, resize, train_subset)

            model = base_model(prior=prior, mu_init=weights_mu_init, device=device,
                               rho_init=weights_rho_init, topology=topology, classes=classes, local_trick=local_trick,
                               sample=sample, **network_parameters, posterior_type=posterior_type)

            model.to(device)

            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Parametri: ', pytorch_total_params)

            if optimizer != 'adam':
                opt = _opt(model.parameters(), lr=lr, momentum=moment)
            else:
                opt = _opt(model.parameters(), lr=lr)

            scheduler = None

            if lr_scheduler is not None:
                t = lr_scheduler.get('type', 'step')

                if t not in LrScheduler:
                    raise ValueError('Supported optimizers', LrScheduler)

                if t == 'step':
                    step, decay = lr_scheduler.get('step', 10), lr_scheduler.get('decay', 0.1)
                    scheduler = torch.optim.lr_scheduler.StepLR(opt, step, gamma=decay)

                elif t == 'exponential':
                    decay = lr_scheduler.get('decay', 0.1)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay)

                elif t == 'plateau':
                    tolerance, decay = lr_scheduler.get('tolerance', 2), lr_scheduler.get('decay', 0.1)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=decay, verbose=True,
                                                                           patience=tolerance, mode='max')

            current_path = os.path.join(save_path, exp_name)  # , str(e))
            current_load_path = os.path.join(load_path, exp_name)

            results = {}
            epoch_start = 0

            results_path = os.path.join(current_path, 'results_{}.data'.format(e))
            best_path = os.path.join(current_path, 'best_model_{}.data'.format(e))

            if save and not os.path.exists(current_path):
                os.makedirs(current_path)

            if load and not os.path.exists(current_load_path):
                os.makedirs(current_load_path)

            loaded = False
            trained = False

            early_stopping = (0, 0)

            if load and os.path.exists(os.path.join(current_load_path, 'results_{}.data'.format(e))):
                loaded = True
                results = torch.load(results_path)
                model.load_state_dict(results['model_state_dict'])
                opt.load_state_dict(results['optimizer_state_dict'])
                epoch_start = results['epoch'] + 1
                early_stopping = results['early_stopping']  # (best_score, early_stopping_tolerance)

                if scheduler is not None:
                    scheduler.load_state_dict(results['optimizer_state_dict'])

                best_score = np.max(results.get('test_results'))

                if load_path != save_path:
                    copy(os.path.join(current_load_path, 'best_model_{}.data'.format(e)), results_path)

            t = trainer(model, train_loader, test_loader, opt, wd=weight_decay)

            if early_stopping[1] < early_stopping_tolerance:
                trained = True
                f1 = results.get('test_results', ['not calculated'])[-1]
                f1_train = results.get('train_results', ['not calculated'])[-1]

                progress_bar = tqdm.tqdm(range(epoch_start, epochs), initial=epoch_start, total=epochs, leave=False)
                progress_bar.set_postfix(f1_test=f1, f1_train=f1_train)

                for i in progress_bar:
                    trained = True

                    if i == 0:
                        (test_true, test_pred) = t.test_evaluation(samples=test_samples)

                        f1 = metrics.f1_score(test_true, test_pred, average='micro')

                        epochs_res = results.get('test_results', [])
                        epochs_res.append(f1)

                        epochs_res_train = results.get('train_results', [])
                        epochs_res_train.append(f1_train)

                        results.update({'epoch': i, 'test_results': epochs_res})

                    start_time = time.time()
                    loss, (train_true, train_pred), (test_true, test_pred) = t.train_step(train_samples=train_samples,
                                                                                          test_samples=test_samples,
                                                                                          weights=loss_weights)
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    loss = np.mean(loss)

                    f1 = metrics.f1_score(test_true, test_pred, average='micro')

                    if scheduler is not None:
                        scheduler.step(f1)

                    f1_train = metrics.f1_score(train_true, train_pred, average='micro')

                    progress_bar.set_postfix(f1_test=f1, f1_train=f1_train)

                    epochs_res = results.get('test_results', [])
                    epochs_res.append(f1)

                    epochs_res_train = results.get('train_results', [])
                    epochs_res_train.append(f1_train)

                    losses = results.get('losses', [])
                    losses.append(loss)

                    training_time = results.get('training_time', 0) + elapsed_time

                    if f1 > best_score:
                        best_score = f1
                        early_stopping = (best_score, 0)
                        if save:
                            torch.save(t.model.state_dict(), best_path)
                    else:
                        early_stopping = (best_score, early_stopping[1] + 1)

                    results.update({
                        'epoch': i, 'test_results': epochs_res, 'train_results': epochs_res_train,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'early_stopping': early_stopping,
                        'training_time': training_time,
                        'scheduler': scheduler.state_dict() if scheduler is not None else None
                    })

                    if save:
                        torch.save(results, results_path)

                    if early_stopping[1] == early_stopping_tolerance:
                        break

                progress_bar.close()
            else:
                run_results.append(results)

            t.model.load_state_dict(torch.load(best_path))
            run_results.append(results)

            # Reliability diagram and ECE

            if not os.path.exists(os.path.join(current_path, "{}_ece.data".format(e))):
                pp, pt, ece, nll = t.reliability_diagram(samples=1)

                r1 = (pp, pt)

                scaled_ece = t.temperature_scaling(samples=1)
                r = (ece, scaled_ece)

                with open(os.path.join(current_path, "{}_ece.data".format(e)), "wb") as f:
                    pickle.dump(r, f)
    
                with open(os.path.join(current_path, "{}_ece_barplot.data".format(e)), "wb") as f:
                    pickle.dump(r1, f)

            print('Max test score {}: {}'.format(seed, np.max(results.get('test_results')[1:])))

            # FGSM ATTACK

            if network != 'normal':

                fgsm_path = os.path.join(current_path, 'fgsm_{}.data'.format(e))

                if not os.path.exists(fgsm_path) and save:
                    r = t.fgsm_test(samples=test_samples)

                    with open(fgsm_path, "wb") as output_file:
                        pickle.dump(r, output_file)

    return experiments_results


if __name__ == '__main__':
    args = sys.argv[1:]

    with open(args[0], "r") as read_file:
        experiments = json.load(read_file)

    results = main(experiments)
