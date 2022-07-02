import numpy as np
import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.utils.data import Dataset
from analysis.net_analysis import analyzeNet
from net import *
import os, sys
from train_sine import *
from analysis.net_analysis import *

initnum = 1
startind = 0
parnums = [0, 3, 4, 8, 12]
eps = [2e-3, 2e-2, 5e-2, 7e-2, 2e-1]
mode_dict = {0: 'nominal', 3: 'nominal_L2', 4: 'lipschitz', 8: 'bayesian', 12:'bayesian_lipschitz'}
parlists = getParlists(act=torch.tanh)
for i in range(startind, initnum):
    train_data = SineDataset(train=True, noise=eps[3], bbb=True)
    test_data = SineDataset(train=False, noise=eps[3], bbb=True)
    data = train_data, test_data

    for p in parnums:
        net = trainNet(*parlists[p], p, data=data)
        net.save('saved_nets/small/' + mode_dict[p] + '/net_sine' + str(p) + '_' + str(i) + '.pt')
        ana = analyzeNet(net)
        torch.save(ana, 'ana_saved/small/' + mode_dict[p] + '/ana_sine' + str(p) + '_' + str(i) + '.pt')
