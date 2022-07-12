# In[]
import numpy as np
import torch
import torchvision
from torch import nn
from train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
from net import *
import sys
from net_analysis import *

# evaluates a single MNIST neural net

def moving_average(data):
    window_width = 20
    cumsum_vec = np.cumsum(np.insert(data, 0, [0]*window_width))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

# In[]
parnum = 8
net = loadNet('d:/trained_nets/saved_nets1/net_mnist' + str(parnum) + '.pt')

# print(analyze(net))
# sys.exit()

print("lmi upper lipschitz bound:", net.lip_const())
# net.saveWeights()

# %%
print("Final train accuracy: ", np.mean(net.train_accuracies[-100:]))
print("Final test accuracy", np.mean(net.test_accuracies[-2:]))

# %%

train_epochs = [i/len(net.train_loader) for i in range(len(net.train_accuracies))]
train_accuracies_smoothed =  moving_average(net.train_accuracies)

plt.plot(train_epochs, train_accuracies_smoothed, label="train accuracy (smoothed)")
plt.plot(net.test_accuracies, label="test accuracy", c='r', zorder=10)
plt.legend()
# plt.yscale('log')
plt.show()

# %%
parnum2 = 1
net2 = loadNet('d:/trained_nets/saved_nets1/net_mnist' + str(parnum2) + '.pt')

plt.plot(net.test_accuracies[:50], net2.test_accuracies[:50]) # plot accuracies against each other
plt.plot([0., 1.], [0., 1.]) #plot diagonal
plt.xlabel("net " + str(parnum) + " accuracy")
plt.ylabel("net " + str(parnum2) + " accuracy")
plt.xlim([0., 1.])
plt.ylim([0., 1.])
plt.show()

# %%

def pred_disagreement(pred1, pred2):
    pred1, pred2 = label(pred1), label(pred2)
    diff = pred1 - pred2
    return np.nonzero(diff)[0].size / len(diff)

n = 20 # min(len(net.test_preds), len(net2.test_preds))
disagreement_matrix = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        disagreement_matrix[i][j] = pred_disagreement(net.test_preds[i], net2.test_preds[j])

plt.matshow(disagreement_matrix)
plt.title("prediction disagreements")
cbar = plt.colorbar(shrink=0.7)
plt.ylabel("net" + str(parnum) + " epochs")
plt.xlabel("net" + str(parnum2) + " epochs")
plt.show()

# %%
# lipschitz distribution
if net.mode == "bayesian":
    def bin_number(x):
        q25, q75 = np.percentile(x, [25, 75])
        bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
        bins = round((x.max() - x.min()) / bin_width)
        return bins

    samples = 100
    lip_consts = [None]*samples

    for i in range(samples):
        net.initWeights()
        lip_consts[i] = net.lip_const(sample=True)
        print(round(100*i/samples), "%", end='\r')

    bins = bin_number(np.array(lip_consts))
    plt.hist(lip_consts, density=True, bins=bins)
    plt.xlabel("lip consts")
    plt.ylabel("probability")
    plt.show()

# verschiedene Anfangsbedingungen -> selbe weights nach training ?
