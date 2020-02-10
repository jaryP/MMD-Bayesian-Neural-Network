import pickle
import time

import matplotlib.pyplot as plt
import torch
import torchvision
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix
from torchvision import datasets
from torchvision.transforms import transforms
import random
from shutil import copy
import ANN
import DropoutNet
from priors import Gaussian, Laplace, ScaledMixtureGaussian, Uniform
from bayesian_layers import BayesianCNNLayer, BayesianLinearLayer
import csv
from itertools import cycle


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


def get_dataset(name, batch_size, dev_split, resize=None):
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
    fig_corr, ax_corr = plt.subplots(nrows=1, figsize=(10,10))
    fig_wro, ax_wro = plt.subplots(nrows=1, figsize=(10,10))

    for d, r in zip(exps, tests):
        if r[0] is None:
            continue

        c, w = [], []

        for i in r:
            c.append(i[0])
            w.append(i[1])

        h = np.asarray(c)
        x = range(h.shape[1])
        stds = np.std(h, 0)
        h = h.mean(0)
        # h = h/h[0]
        ax_corr.plot(x, h, linestyle=d['linestyle'],
                     label='{}'.format(d.get('label')), c=d['color'], linewidth=1.5)
        ax_corr.fill_between(x, h - stds, h + stds, alpha=0.2, color=d['color'])
        ax_corr.set_xticks(x)
        # ax_corr.set_ylim([0, 1])

        h = np.asarray(w)
        stds = np.std(h, 0)
        h = h.mean(0)
        # h = h/h[0]
        ax_wro.plot(x, h, linestyle=d['linestyle'],
                    label='{}'.format(d.get('label')), c=d['color'], linewidth=1.5)
        ax_wro.fill_between(x, h - stds, h + stds, alpha=0.2, color=d['color'])
        ax_wro.set_xticks(x)
        # ax_wro.set_ylim([0, 1])

    handles, labels = ax_corr.get_legend_handles_labels()
    fig_corr.legend(handles, labels, ncol=len(exps), loc='upper center', prop={'size': 25})

    handles, labels = ax_wro.get_legend_handles_labels()
    fig_wro.legend(handles, labels, ncol=len(exps), loc='upper center', prop={'size': 25})

    return (fig_corr, fig_wro), (ax_corr, ax_wro)


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
    adversarial_attack_results = []
    rotation_results = []
    all_reliability = []
    conf_matrices = []

    hists = []
    all_ws = []

    experiments_path = experiment['experiments_path']

    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)

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

        if posterior_type not in PosteriorType:
            raise ValueError('Supported posterior_type', PosteriorType)

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
        local_rotate_res = []
        local_attack_res = []
        local_reliability = []

        for e, seed in tqdm.tqdm(enumerate(seeds), total=len(seeds)):

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            best_score = 0

            if dataset not in Datasets:
                raise ValueError('Supported datasets {}, given {}'.format(Datasets, dataset))
            else:
                sample, classes, train_loader, test_loader = get_dataset(dataset, batch_size, dev_split, resize)

            model = base_model(prior=prior, mu_init=weights_mu_init, device=device,
                               rho_init=weights_rho_init, topology=topology, classes=classes, local_trick=local_trick,
                               sample=sample, **network_parameters, posterior_type=posterior_type)
            # init_model = deepcopy(model)

            model.to(device)

            if optimizer != 'adam':
                opt = _opt(model.parameters(), lr=lr, momentum=moment, weight_decay=weight_decay)
            else:
                opt = _opt(model.parameters(), lr=lr, weight_decay=weight_decay)

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

                # print(results.get('test_results')[1:])
                # print(results.get('train_results'))

                best_score = np.max(results.get('test_results'))

                if load_path != save_path:
                    copy(os.path.join(current_load_path, 'best_model_{}.data'.format(e)), results_path)

            t = trainer(model, train_loader, test_loader, opt)
            trained = False

            if early_stopping[1] < early_stopping_tolerance:
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

            if os.path.exists(os.path.join(current_path, "{}_temp_scaling.data".format(e))) and not trained:
                with open(os.path.join(current_path, "{}_temp_scaling.data".format(e)), "rb") as f:
                    r = pickle.load(f)
            else:
                r1 = t.reliability_diagram(samples=test_samples)
                # r2 = t.temperature_scaling(samples=1)
                r = (r1[-2], r1[-2])
                with open(os.path.join(current_path, "{}_temp_scaling.data".format(e)), "wb") as f:
                    pickle.dump(r, f)

            local_reliability.append(r)

            # # Confusion matrix
            # plt.close('all')
            # true_class, pred_class = t.test_evaluation(samples=test_samples)
            # conf_matrix = confusion_matrix(true_class, pred_class)
            # plt.matshow(conf_matrix)
            # conf_matrices.append(conf_matrix)
            # plt.savefig(os.path.join(current_path, "{}_conf_matrix.pdf".format(e)),
            #             bbox_inches='tight')

            h1 = None
            h2 = None

            if network != 'normal':

                #     if network != 'dropout':
                #         for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                #             t.model.set_mask(p)
                #             (test_true, test_pred) = t.test_evaluation(samples=test_samples)
                #             print(p, metrics.f1_score(test_true, test_pred, average='micro'))
                #
                #     t.model.set_mask(None)

                # attack1_path = os.path.join(current_path, 'attack1_{}.data'.format(e))
                fgsm_path = os.path.join(current_path, 'fgsm_{}.data'.format(e))

                if loaded and not trained:
                    # if os.path.exists(attack1_path):
                    #     with open(attack1_path, "rb") as f:
                    #         h1 = pickle.load(f)

                    if os.path.exists(fgsm_path):
                        with open(fgsm_path, "rb") as f:
                            h2 = pickle.load(f)

                # if h1 is None:
                #     h1 = t.shuffle_test(samples=test_samples)
                #     if save:
                #         with open(attack1_path, "wb") as output_file:
                #             pickle.dump(h1, output_file)

                if h2 is None:
                    h2 = t.fgsm_test(samples=test_samples)
                    if save:
                        with open(fgsm_path, "wb") as output_file:
                            pickle.dump(h2, output_file)

                # Variance matrix
                plt.close('all')

                # plt.figure()
                covar = t.total_variance(samples=test_samples)
                plt.matshow(covar)
                plt.savefig(os.path.join(current_path, "{}_prediction_covariance.pdf".format(e)),
                            bbox_inches='tight')
                # plt.close()

                # a = np.trace(np.dot(conf_matrix.T, covar))
                # b = np.trace(np.dot(conf_matrix.T, conf_matrix))*np.trace(np.dot(covar.T, covar))
                # c = a/np.sqrt(b)
                # print(c)

            local_rotate_res.append(h1)
            local_attack_res.append(h2)

            ws = []
            # ws_init = []
            to_hist = []

            if network in ['bbb', 'mmd']:
                if trained or not os.path.exists(os.path.join(current_path, "{}_{}_posterior.pdf".format(e, network))):
                    # prior = 0
                    # last = None
                    labels = []
                    for layer in t.model.features:
                        wc = []
                        if isinstance(layer, (BayesianLinearLayer, BayesianCNNLayer)):
                            labels.append('CNN' if isinstance(layer, BayesianCNNLayer) else 'Linear')

                            # prior += layer.prior_prob(log=True).item()

                            w = layer.w
                            b = layer.b

                            mean = np.abs(w.mu.detach().cpu().numpy())
                            std = w.sigma.detach().cpu().numpy()
                            snr = mean / std
                            to_hist.append(np.reshape(snr, -1))
                            # ws.append(np.reshape(w.weights.detach().cpu().numpy(), -1))
                            wc.extend(np.reshape(w.weights.detach().cpu().numpy(), -1))
                            if b is not None:
                                mean = np.abs(b.mu.detach().cpu().numpy())
                                std = b.sigma.detach().cpu().numpy()
                                snr = mean / std
                                to_hist.append(np.reshape(snr, -1))
                                # to_hist.append(np.reshape(w.weights.detach().cpu().numpy(), -1))
                                # ws.append(np.reshape(b.weights.detach().cpu().numpy(), -1))
                                wc.extend(np.reshape(b.weights.detach().cpu().numpy(), -1))

                            ws.append(wc)

                    # print(np.exp(prior))
                    # for layer in init_model.features:
                    #     wc = []
                    #     if isinstance(layer, (BayesianLinearLayer, BayesianCNNLayer)):
                    #         w = layer.w
                    #         b = layer.b
                    #
                    #         # ws_init.append(np.reshape(w.weights.detach().cpu().numpy(), -1))
                    #         wc.extend(np.reshape(w.weights.detach().cpu().numpy(), -1))
                    #
                    #         if b is not None:
                    #             # ws_init.append(np.reshape(b.weights.detach().cpu().numpy(), -1))
                    #             wc.extend(np.reshape(b.weights.detach().cpu().numpy(), -1))
                    #
                    #         ws.append(np.asarray(wc))
                    # ws = np.concatenate(ws)
                    # ws_init = np.concatenate(ws_init)

                    to_hist = np.concatenate(to_hist)
                    hists.append(to_hist)
                    all_ws.append(ws)

                    # else:
                    #     hists.append([])
                    #     for m in t.model.parameters():
                    #         ws.append(np.reshape(m.data.detach().cpu().numpy(), -1))
                    #
                    #     # for m in init_model.parameters():
                    #     #     ws_init.append(np.reshape(m.data.detach().cpu().numpy(), -1))
                    #
                    #     # ws = np.concatenate(ws)
                    #     ws_init = np.concatenate(ws_init)
                    plt.close('all')

                    fig, ax = plt.subplots()

                    # ls = cycle(linestyle_tuple)

                    for l, w in enumerate(ws):
                        offset = np.abs(np.min(w) - np.max(w)) * 0.1
                        xs = np.linspace(np.min(w) - offset, np.max(w) + offset, 200)
                        density = gaussian_kde(w)
                        density._compute_covariance()
                        plt.plot(xs, density(xs), label="Layer #{}: {}".format(l + 1, labels[l]), linewidth=1)

                    # xs = np.linspace(ws_init.min(), ws_init.max(), 200)
                    # density = gaussian_kde(ws_init)
                    # density._compute_covariance()
                    # plt.plot(xs, density(xs), label="Initial weights", linewidth=0.5)

                    # if network in ['bbb', 'mmd']:
                    #     s = prior.sample(torch.Size([5000])).numpy()
                    #     xs = np.linspace(s.min(), s.max(), 200)
                    #     density = gaussian_kde(s)
                    #     # # density._compute_covariance()
                    #     plt.plot(xs, density(xs), label="Prior", linewidth=0.5)
                    #     # hist, _, _ = ax.hist(prior.sample(torch.Size([1000])).numpy(), bins=1000, linewidth=0.5, density=False,
                    #     #                      label="Prior", histtype='step')

                    # plt.legend(prop={'size': 10})

                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    fig.savefig(os.path.join(current_path, "{}_{}_posterior.pdf".format(e, network)),
                                bbox_inches='tight')
                    plt.close(fig)

        all_reliability.append(local_reliability)

        rotation_results.append(local_rotate_res)
        adversarial_attack_results.append(local_attack_res)

        # print('-' * 200)
        experiments_results.append(run_results)

    all_reliability = np.asarray(all_reliability)
    # input(reliability)
    experiments = [e for e in experiments if not e.get('skip', False)]

    # fig, ax = plot_test(experiments, rotation_results)
    #
    # # for a in ax:
    # #     a.set_xticklabels(ANN.Trainer.shuffle_percentage, fontsize=8)
    #
    # fig.savefig(os.path.join(experiments_path, "shuffle.pdf"), bbox_inches='tight')
    # plt.close(fig)
    plt.close('all')

    # PLOT FGSM
    fig, ax = plot_test(experiments, adversarial_attack_results)
    # for a in ax:
    ax[0].set_xticklabels(ANN.Trainer.epsilons, fontsize=8)
    ax[1].set_xticklabels(ANN.Trainer.epsilons, fontsize=8)

    ax[0].xaxis.set_tick_params(labelsize=25)
    ax[0].yaxis.set_tick_params(labelsize=25)
    ax[1].xaxis.set_tick_params(labelsize=25)
    ax[1].yaxis.set_tick_params(labelsize=25)

    fig[0].savefig(os.path.join(experiments_path, "fgsm_correct_class.pdf"), bbox_inches='tight')
    fig[1].savefig(os.path.join(experiments_path, "fgsm_wrong_class.pdf"), bbox_inches='tight')

    plt.close('all')

    # Scores plot
    plt.figure(0)

    for i in range(len(experiments)):
        d = experiments[i]
        r = experiments_results[i]
        # ece = reliability[i][-2]
        # nll = reliability[i][-1]

        res = [i['test_results'] for i in r]

        # mx = len(max(_res, key=len))
        # # print('Exp: {}, max score:  {} (at epoch {}). '
        # #       'Training time: {} (s). Ece: {}, Nll: {}'.format(d.get('exp_name'),
        # #                                               np.max(res), np.argmax(res),
        # #                                               r[0].get('training_time', -1), ece, nll))
        #
        # # print('Exp: {}, max score:  {}'.format(d.get('exp_name'), np.max(res)))
        #
        # res = np.zeros((len(_res), mx), dtype=np.float)
        #
        # for i, r in enumerate(_res):
        #     res[i, :len(r)] = r
        #
        # res = np.asarray(res)

        means, stds = unbalanced_mean_std(res)

        # # means = 100 - res.mean(0) * 100
        # # means = res.mean(0)
        # means = np.average(res, weights=(res > 0), axis=0)
        #
        # mask = res > 0
        # stds = np.ma.masked_where(mask, res).std(axis=0)
        # means = means[:np.argmax(means)+1]

        # stds = stds[:np.argmax(means)+1]

        # save_path = d['save_path']
        plt.plot(range(len(means)), means, linestyle=d['linestyle'],
                 label=d.get('label', d['network_type']), c=d['color'], linewidth=2)

        plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.1, color=d['color'])

        # for d, r in zip(experiments, experiments_results):
        #     res = [i['test_results'][1:] for i in r]
        #     print('Exp: {}, max score:  {} (at epoch {})'.format(d.get('exp_name'), np.max(res), np.argmax(res)))
        #     # print('Exp: {}, max score:  {}'.format(d.get('exp_name'), np.max(res)))
        #
        #     res = np.asarray(res)
        #
        #     means = 100 - res.mean(0) * 100
        #     stds = res.std(0)
        #
        #     f1 = means
        #     save_path = d['save_path']train_epoch
        #
        #     plt.plot(range(len(f1)), f1, label=d.get('label', d['network_type']), c=d['color'], linewidth=0.1)
        #
        #     plt.fill_between(range(len(f1)), f1 - stds, f1 + stds, alpha=0.1, color=d['color'])
        #
        #     plt.legend()

        # plt.xticks(range(len(means)), [i + 1 for i in range(len(means))])

    plt.legend()

    plt.ylabel("Test score (%)")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(experiments_path, "score.pdf"), bbox_inches='tight')
    plt.close('all')

    # Results file writing
    with open(os.path.join(experiments_path, 'results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["exp_name", "network", "max_score", "max_score_epoch", "ece", "scaled_ece"])

        for i in range(len(experiments)):
            d = experiments[i]
            r = experiments_results[i]

            ece = all_reliability[i, :, 0] * 100

            # ece = reliability[i][0][-2] * 100
            # new_ece = reliability[i][1][-2] * 100

            # eces = [reliability[i][j][-2] * 100 for j in range(reliability[i])]
            # print(eces)

            res = [i['test_results'] for i in r]
            res, res_std = unbalanced_mean_std(res)
            # res = np.asarray(res) * 100
            # print(res_std)

            res_i = np.argmax(res)

            writer.writerow([d.get('label'), d.get('network_type'),
                             '{} +- {}'.format(res[res_i] * 100, res_std[res_i] * 100),
                             res_i,
                             '{} +- {}'.format(np.mean(ece), np.std(ece)),
                             '{} +- {}'.format(np.mean(ece), np.std(ece))])

            # res = [i['test_results'][1:] for i in r]
            # print('Exp: {}, max score:  {} (at epoch {}). '
            #       'Training time: {} (s). Ece: {}, Nll: {}'.format(d.get('exp_name'),
            #                                               np.max(res), np.argmax(res),
            #                                               r[0].get('training_time', -1), ece, nll))

    # fig, ax = plt.subplots(nrows=1, ncols=1)

    # for d, r in zip(experiments, hists):
    #     if len(r) == 0:
    #         continue
    #
    #     r = 10 * np.log10(r + 1e-12)
    #     xs = np.linspace(r.min() - 10, r.max() + 10, 200)
    #     density = gaussian_kde(r)
    #     density.covariance_factor = lambda: .25
    #     density._compute_covariance()
    #     plt.plot(xs, density(xs), label=d.get('label', d['network_type']), color=d['color'])
    #
    # ax.legend(loc='upper left')
    # fig.savefig(os.path.join(experiments_path, "hists.pdf"), bbox_inches='tight')
    # plt.close(fig)

    # # prob_true, prob_pred = get_diagram_data(y_true, y_prob, 5)
    # #
    # import matplotlib.pyplot as plt

    # fig = plt.figure(1, figsize=(10, 10))

    # ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    # ax2 = plt.subplot2grid((3, 1), (2, 0))

    # ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # ax1.legend(loc='upper left')
    # plt.close('all')
    #
    # fig = plt.figure(1, figsize=(10, 10))
    #
    # ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    # # ax2 = plt.subplot2grid((3, 1), (2, 0))
    #
    # ax1.plot([0, 1], [0, 1], "-", label="Perfectly calibrated", linewidth=1, c='k')
    #
    # for d, ((prob_pred, prob_true, _, _), (prob_pred1, prob_true1, _, _)) in zip(experiments, reliability):
    #     ax1.plot(prob_pred, prob_true, d['linestyle'], label=d.get('label', d['network_type']), c=d['color'],
    #              linewidth=2)
    #     # ax1.plot(prob_pred1, prob_true1, "s--", label=d.get('label', d['network_type'])+'scaled', c=d['color'])
    #
    #     # ax2.hist(prob_true, range=(0, 1), bins=50,
    #     #          histtype="step", lw=2)
    #
    #     # print('Exp: {}, ece: {}'.format(d.get('exp_name'), ece))
    #
    # ax1.legend(loc='upper left')
    #
    # fig.savefig(os.path.join(experiments_path, "calibration_curve.pdf"), bbox_inches='tight')
    # plt.close('all')

    return experiments_results


if __name__ == '__main__':
    import json
    import sys
    import numpy as np

    args = sys.argv[1:]

    with open(args[0], "r") as read_file:
        experiments = json.load(read_file)

    results = main(experiments)
