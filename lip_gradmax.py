import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import functional as fn
from train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
from net import *
import itertools

# estimates lower bound on lipschitz constant by maximizing the squared gradient with respect to inputs
# only feasible for sine dataset

def calcL(net, a=torch.tensor([0.1]).reshape(-1,1), b=torch.tensor([0.9]).reshape(-1,1), m=10):
    lossf = nn.MSELoss()
    def f(x):
        pred = net.forward(x)
        return pred
    def g(x):
        j = fn.jacobian(f, x)
        return torch.square(j)
    def grad_g(x, gx):
        h = fn.hessian(f, x)
        j = torch.sqrt(gx)
        return 2*j*h
    
    ga = g(a)
    gb = g(b)
    ha = grad_g(a, ga)
    hb = grad_g(b, gb)
    if ha*hb > 0:
        return None
    for i in range(m):
        c = (a+b)/2
        gc = g(c)
        hc = grad_g(c, gc)
        if hc*ha > 0:
            a = c
            ga = gc
            ha = hc
        elif hc*hb > 0:
            b = c
            gb = gc
            hb = hc
        else:
            break

    return torch.sqrt(gc).item()

if __name__ == "__main__":
    net = loadNet('d:/trained_nets/saved_nets1/net_sine0.pt')
    l = calcL(net)
    print("\nlower lipschitz bound:", l)
