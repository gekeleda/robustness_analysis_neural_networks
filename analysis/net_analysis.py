import numpy as np
from scipy.stats import describe, gaussian_kde
import torch
from torch import nn
from train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
from net import *
from lip_brute import lip_grid_net

# defines analysis procedure for nets, plus how to plot the finished analysis

n_bay = 10

def moving_average(data):
    window_width = 20
    cumsum_vec = np.cumsum(np.insert(data, 0, [0]*window_width))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

def analyzeNet(net, plot_samples=15, kde_samples=100):
    ana = {}
    ana['mode'] = net.mode
    ana['data_name'] = net.data_name
    ana['epoch_times'] = net.epoch_times

    if net.mode == 'bayesian' or net.mode=='bayesian_lipschitz':
        lips_u = []
        lips_u_s = []
        for i in range(n_bay):
            net.initWeights()
            lips_u.append(net.lip_const(sample=True))
            lips_u_s.append(net.lip_const(spectral=True, sample=True))
        lip_upper = np.mean(lips_u)
        lip_upper_spectral = np.mean(lips_u_s)
        ana['lips_u'] = lips_u
        ana['lips_u_s'] = lips_u_s
    else:
        lip_upper = net.lip_const()
        lip_upper_spectral = net.lip_const(spectral=True)
    train_loss = net.epoch_losses
    test_loss = net.test_losses

    train_aloss = net.epoch_alosses
    test_aloss = net.test_alosses

    ana['lip_upper'] = lip_upper
    ana['lip_upper_spectral'] = lip_upper_spectral
    ana['train_loss'] = train_loss
    ana['test_loss'] = test_loss
    ana['train_aloss'] = train_aloss
    ana['test_aloss'] = test_aloss

    train_loader = net.train_loader
    train_x = train_loader.dataset.data[:, 0]
    train_y = train_loader.dataset.data[:, 1]
    test_loader = net.test_loader
    test_x = test_loader.dataset.data[:, 0]
    test_y = test_loader.dataset.data[:, 1]

    ana['train_x'] = train_x
    ana['train_y'] = train_y
    ana['test_x'] = test_x
    ana['test_y'] = test_y

    if net.data_name == 'sine':

        if net.mode == 'bayesian' or net.mode=='bayesian_lipschitz':
            lips_l = []
            mses = []
            for i in range(n_bay):
                net.initWeights()
                test_pred = net.forward(test_x).detach().numpy()
                test_mse = (np.square(test_pred - test_y.reshape(-1,1).detach().numpy())).mean()
                mses.append(test_mse)
                lips_l.append(lip_grid_net(net))
            ana['mse'] = np.mean(mses)
            ana['mses'] = mses
            lip_lower = np.mean(lips_l)
            ana['lips_l'] = lips_l
        else:
            test_pred = net.forward(test_x).detach().numpy()
            test_mse = (np.square(test_pred - test_y.reshape(-1,1).detach().numpy())).mean()
            ana['mse'] = test_mse
            lip_lower = lip_grid_net(net)

        x_span = np.linspace(-0.25, 1.25, 200).astype(np.float32).transpose()
        x_span = torch.from_numpy(x_span)
        preds = []
        if net.mode == 'bayesian' or net.mode=='bayesian_lipschitz':
            net.model.eval()
            net.setMeans()
            preds.append(net.forward(x_span).detach().numpy())
            for i in range(plot_samples):
                net.initWeights()
                preds.append(net.forward(x_span).detach().numpy())
        else:
            preds.append(net.forward(x_span).detach().numpy())

        ana['lip_lower'] = lip_lower
        ana['x_span'] = x_span
        ana['preds'] = preds

    if net.data_name == 'mnist':
        final_train_acc = np.mean(net.train_accuracies[-100:])

        if net.mode == 'bayesian' or net.mode=='bayesian_lipschitz':
            ft_accs = []
            for i in range(n_bay):
                net.initWeights()
                ft_accs.append(net.accuracy_loader(test_loader))
            final_test_acc = np.mean(ft_accs)
            ana['ft_accs'] = ft_accs
        else:
            final_test_acc = np.mean(net.test_accuracies[-2:])
        train_epochs = [i/len(net.train_loader) for i in range(len(net.train_accuracies))]
        train_accs_smoothed =  moving_average(net.train_accuracies)
        test_accs = net.test_accuracies

        ana['final_train_acc'] = final_train_acc
        ana['final_test_acc'] = final_test_acc
        ana['train_epochs'] = train_epochs
        ana['train_accs_smoothed'] = train_accs_smoothed
        ana['test_accs'] = test_accs

    if net.mode == 'bayesian' or net.mode=='bayesian_lipschitz':
        kde_samples = 100
        lip_consts = [None]*kde_samples

        for i in range(kde_samples):
            net.initWeights()
            lip_consts[i] = net.lip_const(sample=True)
            # print(round(100*i/kde_samples), "%", end='\r')

        description = describe(lip_consts, bias=False)

        kernel = gaussian_kde(lip_consts) # estimate distribution of lip_consts with gaussian kernel density estimation
        spacing = 0.1
        lip_x = np.linspace(min(lip_consts)-spacing, max(lip_consts)+spacing, num=400)
        lip_kde = kernel(lip_x)

        ana['lip_consts'] = lip_consts
        ana['lip_description'] = description
        ana['lip_x'] = lip_x
        ana['lip_kde'] = lip_kde

    return ana

def dispAna(ana):
    # lipschitz constants
    print("brute-force lower lipschitz bound:", ana['lip_lower'])
    print("LMI upper lipschitz bound:", ana['lip_upper'])

    # plot loss
    plt.plot(ana['train_loss'], label="train loss")
    plt.plot(ana['test_loss'], label="test loss", c='r')
    plt.legend()
    # plt.yscale('log')
    plt.show()

    if ana['data_name']=='mnist':
        # print final accuracies
        print("Final train accuracy: ", np.mean(ana['final_train_acc']))
        print("Final test accuracy", np.mean(ana['final_test_acc']))
        # plot accuracy
        plt.plot(ana['train_epochs'], ana['train_accs_smoothed'], label="train accuracy (smoothed)")
        plt.plot(ana['test_accs'], label="test accuracy", c='r', zorder=10)
        plt.legend()
        # plt.yscale('log')
        plt.show()

    if ana['data_name']=='sine':
        fig = plt.figure()
        x_span = np.linspace(-0.25, 1.25, 200).astype(np.float32).transpose()
        x_span = torch.from_numpy(x_span)
        def plotNet(x, preds, datax, datay, samples=15):
            fig.clear()
            alpha = 1.0
            if ana['mode']=='bayesian':
                alpha = 1.0/np.sqrt(0.7*samples)
                # plot mean
                plt.plot(x_span, preds[0], 'yellow', alpha=1.0)
                preds = preds[1:]
            for i in range(len(preds)):
                    plt.plot(x_span, preds[i], 'r', alpha=alpha)

            plt.scatter(datax, datay, s=1, alpha=0.05)
            plt.draw()
            plt.show(block=True)

        plotNet(ana['x_span'], ana['preds'], ana['datax'], ana['datay'], samples=250)

    if ana['mode']=='bayesian':
        # lipschitz distribution
        print(ana['lip_description'])
        plt.plot(ana['lip_x'], ana['lip_kde'])
        plt.xlabel("lip consts")
        plt.ylabel("probability")
        plt.show()


