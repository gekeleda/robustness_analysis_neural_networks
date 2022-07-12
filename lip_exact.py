import numpy as np
import torch
import torchvision
from torch import nn
from train.train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
from net import *
import itertools

# calculates exact lipschitz constant for small 2-layer relu nets with 1d input

eps = 1e-8

def calc_kinks(all_weights, m):
    weights, biases = all_weights[:m], all_weights[m:]
    n = weights[1].shape[0]
    zero_one = [0., 1.]
    zs = list(map(list, itertools.product(zero_one, repeat=n))) # all possible 0-1 vectors of length n
    kinks = []
    rejected = 0
    for i in range(n): # for neuron in first layer
            for z in zs: # for all possible activation states
                if max(z)==0:
                    kinks.append(-biases[1][i]) # manually calculated kink
                s1 = sum([z[j]*weights[1][i][j]*biases[0][j] for j in range(n)])
                s2 = sum([z[j]*weights[1][i][j]*weights[0][j] for j in range(n)])
                if abs(s2) > eps:
                    kink = - (biases[1][i] + s1) / s2 # manually calculated kink
                    kinks.append(kink)
                else:
                    rejected += 1 # reject numerically difficult and irrelevant kinks (to prevent divide by 0)
    print(rejected, "kinks rejected")
    return np.sort(kinks, axis=None) # return all possible kinks a relu net might have

def calcL(net):
    all_weights, m = net.getWeights(with_n=True)
    kinks = calc_kinks(all_weights, m)
    print(kinks)

    y_kinks = net(kinks).detach().numpy() # y value at each kink x 

    x_between = np.array([kinks[0]-1] + [(kinks[i]+kinks[i+1])/2 for i in range(len(kinks)-1)] + [kinks[-1]+1]) # choose one point between neighbouring kinks
    y_between = net(x_between).detach().numpy() # evaluate points between

    slopes = [(y_between[i]-y_kinks[i])/(x_between[i]-kinks[i]) for i in range(len(kinks))] + [(y_between[-1]-y_kinks[-1])/(x_between[-1]-kinks[-1])]
    # calculate all slopes between kinks, which are all possible slopes of network

    # plt.plot(kinks, y_kinks)
    # plt.hist(kinks, bins=500)
    # plt.show()

    return max(np.abs(slopes))[0] # return biggest slope


if __name__ == "__main__":
    net = loadNet('d:/trained_nets/relu/saved_nets3/net_sine1.pt')
    l = calcL(net)
    print("\nexact lipschitz constant:", l)
