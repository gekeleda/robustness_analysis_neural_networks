import numpy as np
import torch
import torchvision
from torch import nn
from train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from net import *
from os import walk
import re
import matplotlib.lines as mlines

# Evaluates different MNIST nets and shows comparison graph

colors = ['red', 'blue', 'green']

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def net_paths(path):
    filenames = next(walk(path), (None, None, []))[2]  # [] if no file

    net_paths = []

    for filename in sorted(filenames, key=numericalSort):
        try:
            if 'mnist' in filename:
                net_paths.append(path + '/' + filename)
        except:
            print(filename)
    return net_paths

act = 'norm'

paths = ['ana_saved/' + act + '/ana_nets1', 'ana_saved/' + act + '/ana_nets2', 'ana_saved/' + act + '/ana_nets3']
netpaths = [net_paths(path) for path in paths]
parlists = [list(x) for x in zip(*netpaths)]

xticks = ['nominal', 'nominal+L2', 'lipschitz', 'bayesian', 'bayesian-lipschitz']

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig.suptitle('classification task: comparison')
ax1.set_xticks([i for i in range(len(parlists))], xticks)
ax1.set_ylabel('test accuracy [-]')
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
accs = []
lips_lmi = []
lips_spectral = []
train_times = []
accs_stds = []
lips_lmi_stds = []
lips_spectral_stds = []

for j in range(len(parlists[0])): # for every intialisation
    indices_tmp = []
    accs_tmp = []
    lips_lmi_tmp = []
    lips_spectral_tmp = []
    train_times_tmp = []
    accs_stds_tmp = []
    lips_lmi_stds_tmp = []
    lips_spectral_stds_tmp = []

    for i in range(len(parlists)): # for every parameter
        ana = torch.load(parlists[i][j])
        final_test_acc = ana['final_test_acc']
        lip_lmi = ana['lip_upper']
        lip_spectral = ana['lip_upper_spectral']
        train_time = ana['epoch_times'][-1]
        indices_tmp.append(i)
        accs_tmp.append(final_test_acc)
        lips_lmi_tmp.append(lip_lmi)
        lips_spectral_tmp.append(lip_spectral)
        train_times_tmp.append(train_time)

        if ana['mode'] == 'bayesian':
            final_test_acc_std = np.std(ana['ft_accs'])
            lip_u_std = np.std(ana['lips_u'])
            lip_u_s_std = np.std(ana['lips_u_s'])
            accs_stds_tmp.append(final_test_acc_std)
            lips_lmi_stds_tmp.append(lip_u_std)
            lips_spectral_stds_tmp.append(lip_u_s_std)
        else:
            accs_stds_tmp.append(0)
            lips_lmi_stds_tmp.append(0)
            lips_spectral_stds_tmp.append(0)
        
        print(round(100*(j*len(parlists) + (i+1))/(len(parlists)*len(parlists[i]))), "%", end="\r")
    
    indices.append(indices_tmp)
    accs.append(accs_tmp)
    lips_lmi.append(lips_lmi_tmp)
    lips_spectral.append(lips_spectral_tmp)
    train_times.append(train_times_tmp)
    accs_stds.append(accs_stds_tmp)
    lips_lmi_stds.append(lips_lmi_stds_tmp)
    lips_spectral_stds.append(lips_spectral_stds_tmp)

offset = [-0.25, 0, 0.25]
alpha = 0.4
width = 0.2
size = 10
print(np.mean(accs, axis=0))
for i in range(len(parlists[0])):
    ax1.scatter(np.asarray(indices[i])+offset[i], accs[i], s=size, marker='x')
    ax2.scatter(np.asarray(indices[i])+offset[i], lips_lmi[i], s=size, marker='x')
for i in range(len(parlists[0])):
    ax2.scatter(np.asarray(indices[i])+offset[i], lips_spectral[i], s=size, marker='o')
    ax3.scatter(np.asarray(indices[i])+offset[i], train_times[i], s=size, marker='x')

# for i in range(len(parlists[0])):
#     if len(accs_stds[i]) > 0:
#         ax1.bar(np.asarray(indices[i])+offset[i], height=2*np.asarray(accs_stds[i]), bottom=np.asarray(accs[i])-np.asarray(accs_stds[i]), alpha=alpha, width=width)
#         ax2.bar(np.asarray(indices[i])+offset[i], height=2*np.asarray(lips_lmi_stds[i]), bottom=np.asarray(lips_lmi[i])-np.asarray(lips_lmi_stds[i]), alpha=alpha, width=width)

ax1.set_ylim([min([min(acc) for acc in accs]) - 0.01, 0.99])
lm = min([min(lip) for lip in lips_lmi])
ax2m = pow(10, np.floor(np.log10(lm)))
ax2.set_ylim([ax2m, max([max(lip) for lip in lips_spectral]) + 1000])
ax3.set_ylim([1e3, 1e5])

for ax in (ax1, ax2, ax3):
        ax.axvline(x=0.5, c='black')
        ax.axvline(x=1.5, c='black')
        ax.axvline(x=2.5, c='black')
        ax.axvline(x=3.5, c='black')

# fig.legend([mpatches.Patch(color=c) for c in colors],
#            ["initialisation " + str(i) for i in range(len(colors))])

cross = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                        markersize=4, label='lmi upper bound')
dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                        markersize=4, label='spectral upper bound')
list_mak = [dot, cross]
# list_lab = ['lmi upper bound', 'spectral norm upper bound']
ax2.legend(handles=list_mak, bbox_to_anchor=(0.4, 0.93), prop={'size':9})

fig.tight_layout()
plt.savefig("figures/comparison_mnist_" + act + ".svg")
plt.show()
