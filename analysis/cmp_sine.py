import numpy as np
import torch
import torchvision
from torch import nn
from train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import matplotlib.patches as mpatches
import matplotlib
from net import *
from os import walk
import re

# Evaluates different regression nets and shows comparison graph

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def sine_paths(path):
    filenames = next(walk(path), (None, None, []))[2]  # [] if no file

    paths = []

    for filename in sorted(filenames, key=numericalSort):
        try:
            if 'sine' in filename:
                paths.append(path + '/' + filename)
        except:
            print(filename)
    return paths

if __name__ == "__main__":

    colors = ['red', 'blue', 'green']

    # scal_formatter = FuncFormatter(lambda y, _: '{:g}'.format(y))
    scal_formatter = ScalarFormatter()

    def limitplot(ax, x, l, u):
        l, u = np.asarray(l), np.asarray(u)
        mean = (l+u)/2
        error = (u-l)/2
        ax.errorbar(x, mean, yerr=error, linestyle="", capsize=3)

    act = 'norm'
    paths = ['ana_saved/' + act + '/ana_nets1', 'ana_saved/' + act + '/ana_nets2', 'ana_saved/' + act + '/ana_nets3']
    netpaths = [sine_paths(path) for path in paths]
    parlists = [list(x) for x in zip(*netpaths)]
    xticks = ['nominal', 'nominal+L2', 'lipschitz', 'bayesian', 'bayesian-lipschitz']

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('regression task: comparison')
    ax1.set_xticks([i for i in range(len(parlists))], xticks)
    ax1.set_ylabel('MSE [-]')
    ax1.set_prop_cycle('color', colors)
    ax1.grid()
    ax1.set_axisbelow(True)

    ax2.set_ylabel('lip. bounds [-]')
    ax2.set_yscale('log')
    ax2.set_prop_cycle('color', colors)
    ax2.grid(True, which='major')
    # ax2.minorticks_on()
    ax2.set_axisbelow(True)

    ax3.set_ylabel('time [s]')
    ax3.set_yscale('log')
    ax3.set_prop_cycle('color', colors)
    ax3.grid()
    ax3.set_axisbelow(True)

    indices = []
    mse_losses = []
    lips_upper = []
    lips_lower = []
    train_times = []
    mse_losses_stds = []
    lips_upper_stds = []
    lips_lower_stds = []

    for j in range(len(parlists[0])): # for every initialisation
        indices_tmp = []
        mse_losses_tmp = []
        lips_upper_tmp = []
        lips_lower_tmp = []
        mse_losses_stds_tmp = []
        lips_upper_stds_tmp = []
        lips_lower_stds_tmp = []
        train_times_tmp = []
        for i in range(len(parlists)): # for every parameter
            ana = torch.load(parlists[i][j])
            mse_loss = ana['mse']
            lip_upper = ana['lip_upper']
            lip_lower = ana['lip_lower']
            train_time = ana['epoch_times'][-1]
            if ana['mode'] == 'bayesian':
                mse_loss_std = np.std(ana['mses'])
                lip_upper_std = np.std(ana['lips_u'])
                lip_lower_std = np.std(ana['lips_l'])
                mse_losses_stds_tmp.append(mse_loss_std)
                lips_upper_stds_tmp.append(lip_upper_std)
                lips_lower_stds_tmp.append(lip_lower_std)
            else:
                mse_losses_stds_tmp.append(0)
                lips_upper_stds_tmp.append(0)
                lips_lower_stds_tmp.append(0)
            indices_tmp.append(i)
            mse_losses_tmp.append(mse_loss)
            lips_upper_tmp.append(lip_upper)
            lips_lower_tmp.append(lip_lower)
            train_times_tmp.append(train_time)
            
            print(round(100*(j*len(parlists) + (i+1))/(len(parlists)*len(parlists[i]))), "%", end="\r")

        indices.append(indices_tmp)
        mse_losses.append(mse_losses_tmp)
        lips_upper.append(lips_upper_tmp)
        lips_lower.append(lips_lower_tmp)
        train_times.append(train_times_tmp)
        mse_losses_stds.append(mse_losses_stds_tmp)
        lips_upper_stds.append(lips_upper_stds_tmp)
        lips_lower_stds.append(lips_lower_stds_tmp)

    offset = [-0.25, 0, 0.25]
    alpha = 0.45
    width = 0.2
    print(np.mean(mse_losses, axis=0))
    for i in range(len(parlists[0])):
        ax1.scatter(np.asarray(indices[i])+offset[i], mse_losses[i], s=15, marker='x')
        # ax2.bar(indices[i], lips_upper[i], bottom=lips_lower[i], color='r', alpha=0.25)
        limitplot(ax2, np.asarray(indices[i])+offset[i], lips_lower[i], lips_upper[i])
        ax3.scatter(np.asarray(indices[i])+offset[i], train_times[i], s=15, marker='x')
        
    
    # for i in range(len(parlists[0])):
    #     if len(mse_losses_stds[i]) > 0:
    #         ax1.bar(np.asarray(indices[i])+offset[i], height=2*np.asarray(mse_losses_stds[i]), bottom=np.asarray(mse_losses[i])-np.asarray(mse_losses_stds[i]), alpha=alpha, width=width)
    #         ax2.bar(np.asarray(indices[i])+offset[i], height=2*np.asarray(lips_upper_stds[i]), bottom=np.asarray(lips_upper[i])-np.asarray(lips_upper_stds[i]), alpha=alpha, width=width)
    # for i in range(len(parlists[0])):
    #     if len(mse_losses_stds[i]) > 0:
    #         ax2.bar(np.asarray(indices[i])+offset[i], height=2*np.asarray(lips_lower_stds[i]), bottom=np.asarray(lips_lower[i])-np.asarray(lips_lower_stds[i]), alpha=alpha, width=width)

        # ax3.scatter(indices[i], lips_lower[i], s=15)

    ax1.set_ylim([0., None])
    lm = min([min(lip) for lip in lips_lower])
    ax2m = pow(10, np.floor(np.log10(lm)))
    ax2.set_ylim([ax2m, None])
    ax3.set_ylim([1, None])

    for ax in (ax1, ax2, ax3):
        ax.axvline(x=0.5, c='black')
        ax.axvline(x=1.5, c='black')
        ax.axvline(x=2.5, c='black')
        ax.axvline(x=3.5, c='black')

    # fig.legend([mpatches.Patch(color=c) for c in colors],
    #         ["initialisation " + str(i) for i in range(len(colors))])

    fig.tight_layout()
    plt.savefig("figures/comparison_sine_" + act + ".svg")
    plt.show()
