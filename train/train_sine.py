#!/usr/bin/env python
# coding: utf-8

# In[7]:

# describes training on regression dataset for several different methods and hyperparameter sets

import numpy as np
from numpy.random import normal
import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.utils.data import Dataset
from net import *
import os, sys

# custom progress bar
class LitProgressBar(TQDMProgressBar):

    def get_metrics(self, trainer, model):
        # don't show the loss (because it is logged in LightningModel already)
        items = super().get_metrics(trainer, model)
        items.pop("loss", None)
        return items

def normal_noise(eps):
    return normal(scale=eps)

class SineDataset(Dataset):
    def __init__(self, train, size=None, noise=0.05, bbb=False):
        if size is None:
            size = 10 if bbb else 10 # big=20, small=10
        if not train:
            size = max(100, size)
        # xvec = np.linspace(0.0, 1.0, num=size)
        xvec = np.random.uniform(size=size)
        if bbb:
            self.bbb = True
            # noise = 0.04 #skalieren!!!
            self.data = torch.from_numpy(np.array([np.asarray([x, 0.5*x + 0.3*np.sin(np.pi*(x+normal_noise(noise))) + 0.3*np.sin(2*np.pi*(x+normal_noise(noise))) + normal_noise(noise)]) for x in xvec]).astype(np.float32))
        else:
            self.bbb = False
            self.data = torch.from_numpy(np.array([np.asarray([x, np.sin(6*x) + noise*torch.normal(torch.Tensor([0.]))]) for x in xvec]).astype(np.float32))
        if not train:
            indices = np.arange(len(self.data))
            indices = np.random.permutation(indices)[:size]
            self.data = self.data[indices]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if np.max(idx) > len(self.data):
            return self[idx-len(self.data)]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx].transpose(0, -1)
        sample = (sample_data[0], sample_data[1])

        return sample

def trainNet(*pars, max_epochs=20000, data=None, bbb=False):

    lr, dims, mode, lr_fac, pretrain_epochs, weight_scale, batch_size, act, act_out, kl_fac, std_scale, sample_size, l, lfac, replace_parameters, l2_fac, parnum = pars

    ### PARAMETERS ######################################################################################################################
    # lr = 1e-3 # None = 1e-3
    # dims = (1, 4, 4, 1)
    # mode = "nominal" # "nominal", "bayesian" or "lipschitz"
    # lr_fac = 0.997
    # pretrain_epochs = 0

    # weight_scale = 0.1
    # batch_size = 256

    # act = act
    # act_out = linear

    # parnum = 0

    # ### BAYESIAN PARAMETERS ###
    # kl_fac = 3e-2
    # std_scale = 3e-3
    # sample_size = 1

    # ### LIPSCHITZ PARAMETERS ###
    # l = 10.0
    # lfac = 3e-3
    # replace_parameters = False

    # ### L2 ###
    # l2_fac = 0.0e-4

    ###################################################################################################################################

    if data is not None:
        train_dataset, test_dataset = data
    else:
        train_dataset = SineDataset(train=True, bbb=bbb)
        test_dataset = SineDataset(train=False, bbb=bbb)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

    loss = nn.MSELoss() # mean squared error loss

    net = Net(dims, mode=mode, regression=True, lr_fac=lr_fac, pretrain_epochs=pretrain_epochs, parnum=parnum, data_name='sine', act=act, act_out=act_out, l2_fac=l2_fac,
        weight_scale=weight_scale, batch_size=batch_size, kl_fac=kl_fac, std_scale=std_scale, sample_size=sample_size,
        l=l, lfac=lfac, replace_parameters=replace_parameters)

    net.configure_lr(lr)
    net.configure_loss(loss)
    net.configure_optimizers()
    net.configure_data(train_loader, test_loader)

    ### Callbacks

    # saves checkpoints to 'net_checkpoints' at every n epochs
    checkpoint_callback = ModelCheckpoint(dirpath='net_checkpoints/', every_n_epochs=500000)

    bar = LitProgressBar()

    # logger
    # logger = TensorBoardLogger("tb_logs", name='baylip', default_hp_metric=False)

    # TRAINER
    trainer = pl.Trainer(max_epochs=max_epochs, logger=False, callbacks=[bar, checkpoint_callback])
    trainer.tune(net)

    #train
    trainer.fit(net, train_loader)

    net.eval()

    net.save()
    return net


def getParlists(act=torch.tanh):
    # act = torch.relu # define activation function
    return [
        # lr , weight dimensions ,  mode   , lr decay, pretrain_epochs, weight_scale, batch_size,   act_func, act_out, kl_fac, std_scale, sample_size, lip_const, lfac, replace_parameters, l2_fac
        [1e-3, (1, 16, 16, 1), 'nominal', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-3, 1, 10.0, 3e-3, False, 0.0],
        [1e-3, (1, 16, 16, 1), 'nominal', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-3, 1, 10.0, 3e-3, False, 0.0],
        [3e-3, (1, 16, 16, 1), 'nominal', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-3, 1, 10.0, 3e-3, False, 0.0],
        [1e-3, (1, 16, 16, 1), 'nominal', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-3, 1, 10.0, 3e-3, False, 9.0e-4],
        [1e-3, (1, 16, 16, 1), 'lipschitz', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-4, 1, 10.0, 3e-3, False, 0.0],
        [3e-3, (1, 16, 16, 1), 'lipschitz', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-3, 1, 10.0, 3e-3, False, 0.0],
        [1e-3, (1, 16, 16, 1), 'lipschitz', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-3, 1, 20.0, 3e-3, False, 0.0],
        [1e-3, (1, 16, 16, 1), 'lipschitz', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-3, 1, 10.0, 3e-2, False, 0.0],
        [1e-3, (1, 16, 16, 1), 'bayesian', 0.9999, 0, 1.0, 1024, act, linear, 1.0e-0, 2e-3, 2, 10.0, 3e-3, False, 0.0],
        [3e-3, (1, 16, 16, 1), 'bayesian', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-3, 1, 10.0, 3e-3, False, 0.0],
        [1e-3, (1, 16, 16, 1), 'bayesian', 0.9999, 0, 1.0, 1024, act, linear, 3e-1, 3e-3, 1, 10.0, 3e-3, False, 0.0],
        [1e-3, (1, 16, 16, 1), 'bayesian', 0.9999, 0, 1.0, 1024, act, linear, 3e-2, 3e-2, 1, 10.0, 3e-3, False, 0.0],
        [1e-3, (1, 16, 16, 1), 'bayesian_lipschitz', 0.9999, 0, 1.0, 1024, act, linear, 0.25e-0, 2e-3, 2, 10.0, 2e-3, False, 0.0],
    ]

if __name__ == '__main__':

    parlists = getParlists(act=torch.tanh)

    if len(sys.argv) > 1:
        parnum = sys.argv[1]
    else:

        # parnum = len(parlists)-1
        parnum = 0

    if parnum == "-1":
        print(len(parlists))
        sys.exit()

    for i in range(len(parlists)):
        parlists[i].append(parnum)

    trainNet(*parlists[int(parnum)], bbb=True, max_epochs=20000)