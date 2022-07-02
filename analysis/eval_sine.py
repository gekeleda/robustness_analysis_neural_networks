# In[]
import numpy as np
from scipy.stats import describe, gaussian_kde, normaltest
import torch
import torchvision
from torch import nn
from train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
from net import *
from lip_brute import lip_grid_net
import sys
from net_analysis import *
# In[]
parnum = 0
net = loadNet('saved_nets/norm/saved_nets1/net_sine0.pt')
# # net = loadNet('saved_nets/net_sine' + str(parnum) + '.pt')
# # net = loadNet('d:/BA_nets/saved_nets/small/lipschitz/net_sine4_19.pt')
# # net.saveWeights()
# epoch_deltas = [net.epoch_times[i+1] - net.epoch_times[i] for i in range(len(net.epoch_times)-1)]
# print("average epoch time:", np.mean(epoch_deltas), "s")

# print(net.lr_fac)

ana = analyzeNet(net)
torch.save(ana, "ana.pt")
ana = torch.load('ana.pt')

# ana = torch.load('d:/BA_nets/ana_saved/small_tmp/bayesian_lipschitz/ana_sine12_2.pt')
# ana = torch.load('ana_saved/norm/ana_nets1/ana_sine12.pt')

# %%
print('Mode:', ana['mode'])
# print(net.dims)
print("MSE:", ana['mse'])
print("brute-force lower lipschitz bound:", ana['lip_lower'])
print("LMI upper lipschitz bound:", ana['lip_upper'])
# %%

# epoch_times = net.epoch_times
epoch_times = [i for i in range(len(ana['train_loss']))]

mintli = np.argmin(ana['test_loss'])
minx = epoch_times[mintli]
plt.plot(epoch_times, ana['train_loss'], label="train loss")
plt.plot(epoch_times, ana['test_loss'], label="test loss", c='r')
plt.axvline(minx, c='black', linestyle='dashed')
plt.xlim(0, min(minx*4, epoch_times[-1]))
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('loss [-]')
plt.yscale('log')
plt.show()

# plt.plot(net.epoch_times, ana['train_loss'], label="train loss")
# plt.plot(net.epoch_times, ana['train_aloss'], label="added loss", c='r')
# plt.legend()
# plt.xlabel('time [s]')
# # plt.yscale('log')
# plt.show()

# %%

fig = plt.figure()
x_span = np.linspace(-0.05, 1.05, 200).astype(np.float32).transpose()
# y_span = np.sin(6*x_span)
y_span = 0.5*x_span + 0.3*np.sin(np.pi*(x_span)) + 0.3*np.sin(2*np.pi*(x_span))
x_span = torch.from_numpy(x_span)
def plotNet(x, preds, trainx, trainy, testx=None, testy=None, samples=15):
    fig.clear()
    plt.xlabel('x [-]')
    plt.ylim(-0.15, 0.9)
    plt.ylabel('y [-]')
    plt.plot(x_span, y_span, label='target function')
    alpha = 1.0
    my_label = 'model prediction'
    if ana['mode']=='bayesian' or ana['mode']=='bayesian_lipschitz':
        alpha = 1.0/np.sqrt(0.25*samples)
        # plot mean
        plt.plot(x_span, preds[0], 'yellow', alpha=1.0, label='mean prediction')
        preds = preds[1:]
    for i in range(len(preds)):
            plt.plot(x_span, preds[i], 'r', alpha=alpha, label=my_label)
            my_label = "_nolegend_"

    scattersize = 3
    plt.scatter(trainx, trainy, s=5*scattersize, alpha=0.99, c='green', label='training data', zorder=100)
    if testx is not None:
        plt.scatter(testx, testy, s=scattersize, alpha=0.99, c='red', label='test data')
        pass
    plt.legend()
    plt.draw()
    plt.show(block=True)

plotNet(ana['x_span'], ana['preds'], ana['train_x'], ana['train_y'], ana['test_x'], ana['test_y'], samples=250)

# %%
if ana['mode']=='bayesian' and False:
    # lipschitz distribution
    print(ana['lip_description'])
    lip_consts = ana['lip_consts']
    z, p = normaltest(lip_consts)
    if p < 0.05:
        print("data is NOT normal distributed, since p =", p, "is smaller 0.05")
    else:
        print("data is approximately normal distributed with p =", p)
    plt.plot(ana['lip_x'], ana['lip_kde'])
    plt.xlabel("lip consts")
    plt.ylabel("probability")
    plt.show()

# %%
