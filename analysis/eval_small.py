import numpy as np
import torch
import torchvision
from torch import nn
from train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
import matplotlib
from net import *
from os import walk
import re
from cmp_sine import sine_paths, numericalSort
from scipy.stats import gaussian_kde
from tqdm import tqdm

# evaluates nets from small regression task in given directories

dirpaths = ['d:/BA_nets/ana_saved/small/' + mode for mode in ['nominal', 'nominal_L2', 'lipschitz', 'bayesian', 'bayesian_lipschitz']]
# dirpaths = ['ana_saved/small/' + mode for mode in ['nominal', 'nominal_L2', 'lipschitz', 'bayesian', 'bayesian_lipschitz']]
anapaths = [sine_paths(path) for path in dirpaths] 

mse_lists = []
lip_lists = []
ana_lists = []
minmse = minlip = 1e10
maxmse = maxlip = 0

for analist in tqdm(anapaths):
    mse_list = []
    lip_list = []
    ana_list = []
    for anapath in analist:
        ana = torch.load(anapath)
        mse = ana['mse']
        mse_list.append(mse)
        lip = ana['lip_lower']
        lip_list.append(lip)
        ana_list.append(ana)
        minmse, minlip = min(minmse, mse), min(minlip, lip)
        maxmse, maxlip = max(maxmse, mse), max(maxlip, mse)
    mse_lists.append(mse_list)
    lip_lists.append(lip_list)
    ana_lists.append(ana_list)

label_dict = {0: 'nominal', 1:'nominal_L2', 2: 'lipschitz', 3: 'bayesian', 4:'bayesian_lipschitz'}
# label_dict = {0: 'nominal', 1: 'lipschitz', 2: 'bayesian', 3:'bayesian_lipschitz'}
# plot
spacing = 0.01
mse_x = np.linspace(minmse-spacing, 0.2, num=400)
lip_x = np.linspace(minlip-spacing, 10, num=400)
means = []
for i in range(len(mse_lists)):
    mses = mse_lists[i]
    means.append(np.mean(mses))
    kernel = gaussian_kde(mses)
    mse_kde = kernel(mse_x)
    plt.plot(mse_x, mse_kde, label=label_dict[i])
print('MSE means:', means)

plt.xlabel('test loss')
plt.legend()
plt.title('test loss distributions from ' + str(len(mse_lists[i])) + ' nets')
plt.xlabel('mse [-]')
plt.ylabel('probability density [-]')

plt.savefig('figures/mse_comp_small.svg')
plt.show()

lmeans = []
for i in range(len(mse_lists)):
    lips = lip_lists[i]
    lmeans.append(np.mean(lips))
    kernel = gaussian_kde(lips)
    lip_kde = kernel(lip_x)
    plt.plot(lip_x, lip_kde, label=label_dict[i])
print('Lipschitz constant means:', lmeans)

plt.legend()
plt.title('lipschitz constant distributions from ' + str(len(mse_lists[i])) + ' nets')
plt.xlabel('lower lipschitz bound [-]')
plt.ylabel('probability density [-]')

plt.savefig('figures/lip_comp_small.svg')
plt.show()

for i in range(len(ana_lists)):
    anas = ana_lists[i]
    # ana = anas[0]
    epoch_timesl = np.array([ana['epoch_times'] for ana in anas])
    test_lossl = np.array(([ana['test_loss'] for ana in anas]))
    epoch_times = np.mean(epoch_timesl, axis=0)
    test_loss = np.mean(test_lossl, axis=0)
    plt.plot(epoch_times, test_loss, label=label_dict[i], zorder=10-i)
plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('test loss [-]')

plt.savefig('figures/time_comp_small.svg')
plt.show()
