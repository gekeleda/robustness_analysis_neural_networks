#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
import matplotlib.pyplot as plt
from torch.distributions import Normal
from net import *
import os, sys

# custom progress bar
class LitProgressBar(TQDMProgressBar):

    def get_metrics(self, trainer, model):
        # don't show the loss (because it is logged in LightningModel already)
        items = super().get_metrics(trainer, model)
        items.pop("loss", None)
        return items

def trainNet(*pars, max_epochs=10000):

    lr, dims, mode, lr_fac, pretrain_epochs, weight_scale, batch_size, act, act_out, kl_fac, std_scale, sample_size, l, lfac, replace_parameters, l2_fac, parnum = pars

    ### PARAMETERS ######################################################################################################################
    # lr = 1e-3 # None = 1e-3
    # dims = (196, 100, 30, 10)
    # mode = "nominal" # "nominal", "bayesian" or "lipschitz"
    # lr_fac = 0.997
    # pretrain_epochs = 0

    # weight_scale = 0.1
    # batch_size = 1024

    # act = torch.tanh
    # act_out = softmax

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

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(os.getcwd(), train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),              
                                    torchvision.transforms.Resize((14, 14)),
                                    torchvision.transforms.Lambda(torch.flatten)
                                ]),
                                target_transform=torchvision.transforms.Lambda(onehot)
                                ),
                                shuffle=True, batch_size=batch_size)

    mnist_test = torchvision.datasets.MNIST(os.getcwd(), train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),              
                                    torchvision.transforms.Resize((14, 14)),
                                    torchvision.transforms.Lambda(torch.flatten)
                                ]),
                                target_transform=torchvision.transforms.Lambda(onehot)
                                )                          
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=len(mnist_test))


    loss = nn.CrossEntropyLoss() # cross entropy loss

    net = Net(dims, mode=mode, lr_fac=lr_fac, pretrain_epochs=pretrain_epochs, parnum=parnum, data_name='mnist', act=act, act_out=act_out, l2_fac=l2_fac,
        weight_scale=weight_scale, batch_size=batch_size, kl_fac=kl_fac, std_scale=std_scale, sample_size=sample_size,
        l=l, lfac=lfac, replace_parameters=replace_parameters)

    net.configure_lr(lr)
    net.configure_loss(loss)
    net.configure_optimizers()
    net.configure_data(train_loader, test_loader)

    ### Callbacks

    # saves checkpoints to 'net_checkpoints' at every n epochs
    checkpoint_callback = ModelCheckpoint(dirpath='net_checkpoints/', every_n_epochs=5000000)

    bar = LitProgressBar()

    # logger
    # logger = TensorBoardLogger("tb_logs", name='baylip', default_hp_metric=False)

    # TRAINER
    trainer = pl.Trainer(max_epochs=max_epochs, logger=False, callbacks=[bar, checkpoint_callback])
    # trainer.tune(net)

    #train
    trainer.fit(net, train_loader)

    net.eval()
    net.save()

if __name__ == '__main__':

    act = torch.tanh # define activation function

    if len(sys.argv) > 1:
        parnum = sys.argv[1]
    else:
        parnum = 12

    # header von parameter tabelle machen ODER dictionary ODER data class
    parlists = [
        # lr , weight dimensions ,     mode   , lr decay, pretrain_epochs, weight_scale, batch_size, act, act_out, kl_fac, std_scale, sample_size, lip_const, lfac, replace_parameters, l2_fac
        [1e-3, (196, 100, 30, 10),   'nominal',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-3,           1,      10.0, 3e-3,              False,  0.0],
        [1e-3, (196, 100, 30, 10),   'nominal',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-3,           1,      10.0, 3e-3,              False,  0.0],
        [3e-3, (196, 100, 30, 10),   'nominal',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-3,           1,      10.0, 3e-3,              False,  0.0],
        [1e-3, (196, 100, 30, 10),   'nominal',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-3,           1,      10.0, 3e-3,              False,  5.0e-4],
        [1e-3, (196, 100, 30, 10), 'lipschitz',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-3,           1,      15.0, 3e-3,              False,  0.0],
        [3e-3, (196, 100, 30, 10), 'lipschitz',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-3,           1,      15.0, 3e-3,              False,  0.0],
        [1e-3, (196, 100, 30, 10), 'lipschitz',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-3,           1,      50.0, 3e-3,              False,  0.0],
        [1e-3, (196, 100, 30, 10), 'lipschitz',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-3,           1,      15.0, 3e-2,              False,  0.0],
        [1e-3, (196, 100, 30, 10),  'bayesian',    0.9995,               0,          0.1,       1024, act, softmax,   1.0e-0,    1.5e-3,         2,      10.0, 3e-3,              False,  0.0],
        [3e-3, (196, 100, 30, 10),  'bayesian',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-3,           1,      10.0, 3e-3,              False,  0.0],
        [1e-3, (196, 100, 30, 10),  'bayesian',    0.9995,               0,          0.1,       1024, act, softmax,   3e-1,      3e-3,           1,      10.0, 3e-3,              False,  0.0],
        [1e-3, (196, 100, 30, 10),  'bayesian',    0.9995,               0,          0.1,       1024, act, softmax,   3e-2,      3e-2,           1,      10.0, 3e-3,              False,  0.0],
        [1e-3, (196, 100, 30, 10), 'bayesian_lipschitz', 0.9995,         0,          1.0,       1024, act, softmax,   1.0e-0,    1.5e-3,         2,      15.0, 3e-3,              False,  0.0]
    ]

    if parnum == "-1":
        print(len(parlists))
        sys.exit()

    parlist = parlists[int(parnum)]
    parlist.append(parnum)

    trainNet(*parlist, max_epochs=1000)
