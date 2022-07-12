import numpy as np
import torch
import torchvision
from torch import nn
from train.train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
from net import *

num = 1000
dist = 5

def lip_point(f, a, b):
    return np.linalg.norm(f(b)-f(a))/np.linalg.norm(b-a) # lower bound for global lipschitz constant

def lip_grid(f, a):
    n = len(a)
    coords = np.array([np.linspace(a[i]-dist, a[i]+dist, num=num) for i in range(n)])
    mesh_coords = np.meshgrid(*coords) # create grid of input domain
    points = np.transpose(np.vstack(list(map(np.ravel, mesh_coords))))
    lbs = [lip_point(f, a, b) for b in points] # calculate slope for all points
    imax = max(range(len(lbs)), key=lbs.__getitem__)
    return lbs[imax] # return biggest slope

def lip_grid_net(net):
    # returns lower bound of lipschitz constant of net
    a = np.zeros((net.dims[0])) + 0.5 # 0.5 is middle of dataset
    f = lambda x: net.forward(x).detach().numpy()
    return lip_grid(f, a)

if __name__ == "__main__":

    # def f(x):
    #     return np.sin(x[0]) + np.cos(2*x[1])
    # a = np.array([1., 2.])
    # print(lip_grid(f, a))

    parnum = 1
    net = loadNet('d:/trained_nets/tanh/saved_nets1/net_sine' + str(parnum) + '.pt')
    print(lip_grid_net(net))

# besser um a=0.5 berechnen!
