import numpy as np
import torch
import torchvision
from torch import nn
from train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import matplotlib.patches as mpatches
import matplotlib
from net import *
from os import walk
import re
from net_analysis import *
from tqdm import tqdm

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def analyze_paths(path, num, anapath):
    filenames = next(walk(path), (None, None, []))[2]  # [] if no file

    for filename in tqdm(sorted(filenames, key=numericalSort)):
        if 'mnist' in filename:
            continue
        try:
            fullpath = path + '/' + filename
            net = loadNet(fullpath)
            ana = analyzeNet(net)
            torch.save(ana, anapath + '/' + 'ana' + filename[3:])
        except Exception as e:
            print(repr(e))
            print(filename)
            return

def edit_ana_paths(netpath, anapath):
    filenames = next(walk(netpath), (None, None, []))[2]  # [] if no file

    net_paths = []

    for filename in sorted(filenames, key=numericalSort):
        try:
            fullnetpath = netpath + '/' + filename
            fullanapath = anapath + '/' + filename
            net = loadNet(fullnetpath)
            ana = torch.load(fullanapath)
            
            if net.mode=='bayesian' and net.data_name=='sine' and net.parnum in ['8', '9', '10']:
                kde_samples = 10000
                lip_consts = [None]*kde_samples

                for i in range(kde_samples):
                    net.initWeights()
                    lip_consts[i] = net.lip_const(sample=True)
                    print(round(100*i/kde_samples), "%", end='\r')

                description = describe(lip_consts, bias=False)

                kernel = gaussian_kde(lip_consts)
                spacing = 0.1
                lip_x = np.linspace(min(lip_consts)-spacing, max(lip_consts)+spacing)
                lip_kde = kernel(lip_x)

                ana['lip_consts'] = lip_consts
                ana['lip_description'] = description
                ana['lip_x'] = lip_x
                ana['lip_kde'] = lip_kde

            torch.save(ana, fullanapath)
        except:
            print(fullanapath)
            return False
    return True

pathnums = ['1', '2', '3']
netpaths = ['saved_nets/norm/saved_nets' + pathnum for pathnum in pathnums]
anapaths = ['ana_saved/norm/ana_nets' + pathnum for pathnum in pathnums]

# pathnums = ['1', '2', '3', '4', '5']
# net_modes = ['nominal', 'lipschitz', 'bayesian', 'bayesian_lipschitz', 'nominal_L2']
# netpaths = ['saved_nets/small_old/' + net_mode for net_mode in net_modes]
# anapaths = ['ana_saved/small/' + net_mode for net_mode in net_modes]

netpaths = [analyze_paths(netpaths[i], pathnums[i], anapaths[i]) for i in range(len(netpaths))]
# automatically analyze all nets in given directories