from unittest import skip
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.autograd import Function
import copy
from lip_cvx import solveLipSDP
import time

meps = 1e-10
eps = 1e-5
startdiff = 1e-9 * 5.0e4
epochfac = 1e-2

class PLogDet(Function):
    @staticmethod
    def forward(ctx, x, l=None):
        if l is None:
            l, info = torch.linalg.cholesky_ex(x)
            if info > 0:
                raise ValueError(('Matrix is not positive definite (leading minor: ' + str(info.detach().item()) + ')'))
        ctx.save_for_backward(l)
        return 2 * l.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    @staticmethod
    def backward(ctx, g):
        l, = ctx.saved_tensors
        # use cholesky_inverse
        return g * torch.cholesky_inverse(l), None
plogdet = PLogDet.apply

def onehot(y):
    return torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

def label(y):
    if torch.is_tensor(y):
        return torch.argmax(y, dim=1)
    else:
        return np.argmax(y, axis=1)

def spectral_norm(weights):
    log_l = 0 # sum log of lipschitz constants instead of multiplying constants
    for weight in weights:
        log_l += np.log(np.linalg.norm(weight, ord=2))
    return np.exp(log_l)

def isPD(Q):
    l, info = torch.linalg.cholesky_ex(Q)
    return info <= 0

def relu(x):
    return torch.relu(x)

lrelu = torch.nn.LeakyReLU(0.1)

def linear(x):
    return x

def sigmoid(x):
    return torch.sigmoid(x)

def tanh(x):
    return torch.tanh(x)

def softmax(x):
    return nn.functional.softmax(x, dim=1)

def softplus(x):
    return nn.functional.softplus(x)

def softplusinv(x):
    x = torch.tensor(x)
    return torch.log(torch.exp(x)-1)

def weightParameter(shape, weight_scale):
    mean = torch.zeros(shape)
    std = torch.ones(shape)*weight_scale
    return torch.nn.Parameter(torch.normal(mean, std))

def rhoParameter(shape, rho_mean):
    mean = torch.zeros(shape) + rho_mean
    std = torch.ones(shape)
    return torch.nn.Parameter(torch.normal(mean, std))

def sampleParameter(mean, std, shape):
    st_mean = torch.zeros(shape)
    st_std = torch.ones(shape)
    return mean + std * torch.normal(st_mean, st_std)

def baseParameter(mean, std, shape):
    st_mean = torch.zeros(shape) + mean
    st_std = torch.ones(shape)*std
    return torch.nn.Parameter(torch.normal(st_mean, st_std))

def loadNet(path):
    tmpnet = Net(empty=True)
    net = tmpnet.load(path)
    net.eval()
    return net

class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(
                self,
                model,
                optimizer,
                lr
        ) -> None:
            self.model = model
            self.update_steps = 0
            self.lr = lr
            super().__init__(optimizer)

        def set_lr(self, optimizer, lr):
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            self.model.lr = self.lr

        def step(self, fac=0.7, cpstep=False):
            if self.lr >= meps and self.lr*fac < meps:
                return self.lr
                self.lr = self.lr*fac
                self.set_lr(self.optimizer, self.lr)
                # raise KeyboardInterrupt
            self.lr = self.lr*fac
            self.set_lr(self.optimizer, self.lr)
            if cpstep:
                self.update_steps += 1
                print(" cp step " + str(self.update_steps))
                print(" lr = " + str(self.lr))
            return self.lr

class Net(pl.LightningModule):
    def __init__(self, dims=(1,1), act=tanh, act_out=linear, weight_scale=1.0, batch_size=8, mode='nominal', regression=False, 
    lr_fac=0.997, pretrain_epochs=0, l2_fac=None, parnum=0, data_name="", empty=False, loading=False, #standard parameters
    kl_fac=1.0, noise=0.1, std_scale=0.07, sample_size=5, # bayesian parameters
    l=1.0, lfac=1e-2, replace_parameters=False, diff=startdiff, beta=5.0, max_cpsteps=5): # lipschitz parameters
        if empty:
            return
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization=False
        self.pretrain_epochs = pretrain_epochs
        self.model = NetModel(dims, act, act_out, weight_scale, batch_size, mode, pretrain_epochs, #standard parameters
            kl_fac, noise, std_scale, sample_size, #bayesian
            l2=l**2, lfac=lfac, replace_parameters=replace_parameters, #lipschitz
            loading=loading)
        self.dims = dims
        self.batch_size = batch_size
        self.mode = mode
        self.regression = regression
        self.parnum = parnum
        self.data_name = data_name
        self.Q, self.L = None, None
        self.lmod = False
        self.diff = diff
        self.beta = beta
        self.max_cpsteps = max_cpsteps
        self.cpsteps = 0
        self.last_loss = None
        self.lr_fac = lr_fac
        self.future_mode = mode
        if pretrain_epochs > 0:
            self.mode = "nominal"
            self.model.mode = "nominal"
            self.l2_fac = 0
        else:
            self.l2_fac = l2_fac
        self.l2_fac_end = l2_fac

        self.losses = []
        self.alosses = []
        self.epoch_losses = []
        self.epoch_alosses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_losses = []
        self.test_alosses = []
        self.test_preds = []
        self.epoch_times = []

        # self.register_buffer("losses", [])

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        lsum = self.calc_batch_loss(train_batch)

        opt.zero_grad()
        self.manual_backward(lsum)

        model_state = copy.deepcopy(self.model.state_dict()) # save the model state
        opt_state = copy.deepcopy(opt.state_dict()) # save the opt state
        opt.step()
        if self.mode == "lipschitz" or self.mode == 'bayesian_lipschitz':
            Q = self.calc_Q()
            Q = Q + eps*torch.eye(len(Q))
            l, info = torch.linalg.cholesky_ex(Q)
            step_performed = False
            while info > 0:
                step_performed = True
                self.model.load_state_dict(model_state)
                opt.load_state_dict(opt_state)
                sch.step() # decrease learning rate
                opt.step()
                Q = self.calc_Q() + eps*torch.eye(len(self.Q))
                l, info = torch.linalg.cholesky_ex(Q)
            if step_performed:
                lsum = self.calc_batch_loss(train_batch)
            self.Q, self.L = Q, l


        if (self.mode == "lipschitz" or self.mode == 'bayesian_lipschitz') and not self.lmod and not self.last_loss is None and abs(lsum - self.last_loss) < self.diff:
            self.cpstep()
            self.lmod = True
        else:
            self.lmod = False

        self.last_loss = lsum.detach().item()
        return lsum

    def calc_batch_loss(self, train_batch, test=False):
        lossf = self.lossf
        x, y = train_batch[0], train_batch[1]

        loss = self.calc_loss((x, y), lossf, test=test)
        return loss

    def calc_loss(self, data, lossf, test=False):
        x, y = data
        # x = x.reshape(-1)
        if self.regression:
            y = y.reshape(-1, 1)

        # Compute prediction error
        if self.mode == 'bayesian' or self.mode == 'bayesian_lipschitz':
            lsum = 0
            asum = 0
            for i in range(self.model.sample_size):
                pred = self.forward(x)
                loss = lossf(y, pred)
                aloss = self.complexity_cost()
                if self.mode == 'bayesian_lipschitz':
                    aloss += self.lipschitz_cost()
                lsum = lsum + loss
                asum = asum + aloss
            lsum, asum = lsum/self.model.sample_size, asum/self.model.sample_size
            tsum = lsum + asum
            if self.l2_fac is not None:
                tsum = tsum + self.l2_cost()

            if not self.regression and not test:
                train_accuracy = self.accuracy_batch(x, y, pred)
                self.train_accuracies.append(train_accuracy)
                self.log("train_accuracy", self.train_accuracies[-1], on_step=True, on_epoch=False, prog_bar=True, logger=False)

            if not test:
                self.log("train_loss", tsum, on_step=True, on_epoch=False, prog_bar=True, logger=False)
                self.losses.append(lsum.detach().item())
                self.alosses.append(asum.detach().item())
                return tsum
            else:
                return lsum, asum

        pred = self.forward(x)
        loss = lossf(y, pred)

        if not self.regression and not test:
            train_accuracy = self.accuracy_batch(x, y, pred)
            self.train_accuracies.append(train_accuracy)
            self.log("train_accuracy", self.train_accuracies[-1], on_step=True, on_epoch=False, prog_bar=True, logger=False)
        
        aloss = 0
        if self.mode == 'lipschitz' or self.mode == 'bayesian_lipschitz':
            aloss += self.lipschitz_cost()
        if self.l2_fac is not None:
            aloss += self.l2_cost()
        tloss = loss + aloss
        
        if not test:
            self.log("train_loss", tloss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
            self.losses.append(loss.detach().item())
            self.alosses.append(aloss.detach().item())
            return tloss
        else:
            return loss, aloss

    def training_epoch_end(self, outputs):
        if len(self.epoch_losses) == 0:
            self.t0 = time.time()
        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean().detach().item()
        avg_loss = np.mean(self.losses[-len(self.train_loader):])
        self.epoch_losses.append(avg_loss)
        avg_aloss = np.mean(self.alosses[-len(self.train_loader):])
        self.epoch_alosses.append(avg_aloss)

        if self.regression:
            test_loss, test_aloss = self.calc_loader_loss(self.test_loader)
            test_loss, test_aloss = test_loss.detach().item(), test_aloss.detach().item()
            self.test_losses.append(test_loss)
            self.test_alosses.append(test_aloss)
            self.log("test_loss", self.test_losses[-1] + self.test_alosses[-1], on_step=False, on_epoch=True, prog_bar=True, logger=False)
        else:
            test_accuracy = self.accuracy_loader(self.test_loader)
            self.test_accuracies.append(test_accuracy)
            self.log("test_accuracy", self.test_accuracies[-1], on_step=False, on_epoch=True, prog_bar=True, logger=False)

        if len(self.epoch_losses) > 1 and (self.mode == 'lipschitz' or self.mode == 'bayesian_lipschitz') and abs(self.epoch_losses[-1] - self.epoch_losses[-2]) < self.diff*epochfac:
            self.cpstep()

        if self.pretrain_epochs < 1:
            self.l2_fac = self.l2_fac_end
        elif self.pretrain_epochs > len(self.epoch_losses):
            self.l2_fac = self.l2_fac_end * np.min([len(self.epoch_losses)/self.pretrain_epochs, 1])
        if len(self.epoch_losses) == self.pretrain_epochs:
            if self.future_mode != 'nominal':
                self.l2_fac = 0
            self.mode = self.future_mode
            self.model.mode = self.future_mode
            if self.future_mode == 'lipschitz' or self.future_mode == 'bayesian_lipschitz':
                # l = self.model.setSDPlambdas(replace_pars=False)
                if self.model.replace_parameters:
                    ldiags = [torch.diag(lam) for lam in self.model.lambdas]
                    means = [torch.nn.Parameter(torch.mm(ldiags[i], self.model.means[i])) for i in range(len(ldiags))]
                    self.model.means[:self.model.n-1] = means
                # print(" Switching to lipschitz mode with lip const. of: ", l)

        if self.mode != "lipschitz" and self.mode != 'bayesian_lipschitz':
            sch = self.lr_schedulers()
            sch.step(self.lr_fac)

        if len(self.epoch_losses) % 25 == 0:
            # torch.save(self, 'saved_nets/net_' + self.data_name + str(self.parnum) + '.pt')
            self.save()
        
        self.epoch_times.append(time.time()-self.t0)

    def forward(self, x, skip_last=False):
        if not torch.is_tensor(x):
            x = torch.tensor(x.astype(np.float32))
        if self.model.training:
            self.initWeights()
        return self.model(x, skip_last=skip_last)

    def configure_lr(self, lr=1e-3):
        self.lr = lr

    def configure_optimizers(self):
        if self.lr is None or 0.5*self.lr < meps:
            self.lr = 1e-3

        # self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, nesterov=True)
        self.optim = torch.optim.NAdam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-7)

        return [self.optim], [{"scheduler":LRScheduler(self, self.optim, self.lr), "interval": "step"}]

    def configure_loss(self, lossf):
        self.lossf = lossf

    def configure_data(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

        if not self.regression:
            # train_accuracy = self.accuracy_loader(self.train_loader).detach().item()
            # self.train_accuracies.append(train_accuracy)
            # self.log("train_accuracy", self.train_accuracies[-1], on_step=False, on_epoch=True, prog_bar=True, logger=False)
            test_accuracy = self.accuracy_loader(self.test_loader)
            self.test_accuracies.append(test_accuracy)
            self.log("test_accuracy", self.test_accuracies[-1], on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def initWeights(self):
        self.model.initWeights()
    
    def setMeans(self):
        self.model.setMeans()

    def getWeights(self, with_n):
        return self.model.getWeights(with_n=with_n)
    
    def getWeightsOnly(self, as_tensors=True, sample=False):
        return self.model.getWeightsOnly(as_tensors=as_tensors, sample=sample)
    
    def saveWeights(self):
        if self.mode == 'bayesian' or self.mode == 'bayesian_lipschitz':
            self.setMeans()
        w = np.array(self.getWeightsOnly(as_tensors=False), dtype=object)
        np.save("weights.npy", w)

    def save(self, path=None):
        if path is None:
            path = 'saved_nets/net_' + self.data_name + str(self.parnum) + '.pt'
        if self.trainer is not None:
            opt = self.optimizers()
        else:
            opt = self.opt
        state = {
            'state_dict': self.state_dict(),
            'optim': opt.state_dict(),
        }

        for attribute, value in self.__dict__.items():
            if attribute not in ["optim", "model", "trainer"] and attribute[0] != "_":
                state[attribute] = value
        state['model_state'] = self.model.getState()

        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        dims = state['dims']
        mode = state['mode']
        self = Net(dims, mode=mode, loading=True)
        self.load_state_dict(state['state_dict'])

        for attribute, value in state.items():
            if attribute != "state_dict" and attribute != "optim" and attribute != "model_state":
                setattr(self, attribute, value)

        self.configure_optimizers()
        opt = self.optim
        self.opt  = opt
        opt.load_state_dict(state['optim'])
        self.model = self.model.restoreState(state['model_state'])

        return self

    def l2_cost(self):
        wvec = self.model.getWeightsVector()
        return self.l2_fac * torch.norm(wvec)**2

    def complexity_cost(self):
        return self.model.kl_weight*(self.var_log_prob() - self.prior_log_prob())
    
    def var_log_prob(self):
        res = 0
        for i in range(len(self.model.means)):
            for j in range(len(self.model.means[i])):
                dist = Normal(self.model.means[i].flatten()[j], softplus(self.model.rhos[i].flatten()[j]))
                res += dist.log_prob(self.model.weights[i].flatten()[j])
        return res/sum(self.model.dims)

    def prior_log_prob(self):
        res = 0
        for i in range(len(self.model.means)):
            for j in range(len(self.model.means[i])):
                dist = Normal(torch.zeros((1)), self.model.weight_scale)
                res += dist.log_prob(self.model.weights[i].flatten()[j])
        return res/sum(self.model.dims)

    def lipschitz_cost(self):
        if self.Q is None:
            self.Q = self.calc_Q()
            self.Q = self.Q + eps*torch.eye(len(self.Q))
            ldet = plogdet(self.Q)
        else:
            ldet = plogdet(self.Q, self.L)
        return -self.model.lfac * ldet

    def calc_Q(self):
        return self.model.calc_Q()

    def cpstep(self):
        fac = 0.99
        cpfac = 0.85
        sch = self.lr_schedulers()
        sch.step(fac, cpstep=True) # decrease learning rate
        if self.lr > meps:
            self.model.lfac *= cpfac # decrease parameter to better approximate real barrier function
            self.diff /= self.beta # decrease stopping criterion diff
            print(" lfac = " + str(self.model.lfac))
            self.cpsteps += 1
        # if self.cpsteps >= self.max_cpsteps:
        #     print("maximal amount of cpsteps reached")
        #     raise KeyboardInterrupt
        self.lmod = True

    def lip_const(self, spectral=False, sample=False):
        dims = self.dims
        if spectral:
            means = self.getWeightsOnly(as_tensors=False, sample=sample)
            return spectral_norm(means)
        else:
            means = self.getWeightsOnly(as_tensors=True, sample=sample)
            if self.model.replace_parameters:
                ldiags = [torch.diag(lam) for lam in self.model.lambdas]
                means = [torch.mm(torch.inverse(ldiags[i]), means[i]) for i in range(len(ldiags))] + [means[-1]]
            l, lambdas = solveLipSDP([mean.detach().numpy() for mean in means])
            return l

    def calc_loader_loss(self, loader):
        iloader = enumerate(loader)

        lsum = 0
        asum = 0
        batch_idx = -10
        while batch_idx < len(loader)-1:
            batch_idx, batch = next(iloader)
            lc, ac = self.calc_batch_loss(batch, test=True)
            lsum += lc
            asum += ac

        return lsum/len(loader), asum/len(loader)

    def accuracy_loader(self, loader):
        if self.regression:
            print("Accuracy cannot be calculated for regression task!")
            raise KeyboardInterrupt

        iloader = enumerate(loader)

        correct = total = 0
        batch_idx = -10
        while batch_idx < len(loader)-1:
            batch_idx, (x, y) = next(iloader)
            pred = self.forward(x)
            if loader is self.test_loader:
                self.test_preds.append(pred.detach().numpy())
            correct_batch, total_batch = self.correct_in_batch(x, y, pred=pred)
            correct += correct_batch
            total += total_batch

        return correct/total

    def correct_in_batch(self, x, y, pred=None):
        correct = 0
        total = len(x)
        if pred == None:
            pred = self.forward(x)
        y, pred = label(y), label(pred)
        correct = torch.sum(y == pred)
        return correct.item(), total

    def accuracy_batch(self, x, y, pred=None):
        c, t = self.correct_in_batch(x, y, pred=pred)
        return c/t
        
        

class NetModel(nn.Module):
    def __init__(self, dims, act=tanh, act_out=linear, weight_scale=1.0, batch_size=1, mode='nominal', pretrain_epochs=0,
        kl_fac=1.0, noise=0.1, std_scale=0.05, sample_size=5,
        l2=1.0, lfac=1e-3, replace_parameters=False, loading=False):
        super(NetModel, self).__init__()
        self.pretrain_epochs = pretrain_epochs
        self.dims = dims
        self.n = len(dims)-1
        self.act = act
        self.act_out = act_out
        self.weight_scale = weight_scale
        self.batch_size = batch_size
        self.mode = mode
        self.weights = None
        self.replace_parameters = replace_parameters
        self.lambdas = None
        self.loading = loading

        if mode == 'lipschitz' or mode == 'bayesian_lipschitz':
            self.l2 = l2
            self.lfac = lfac

            fac = 0.9
            lambda_mean = 10.0
            lambda_std = 0.1
            # self.weight_scale = weight_scale*min(1.0, fac*np.sqrt(l2)*self.n/(sum(self.dims))) # divide l by average neuron count per layer
            self.means = [weightParameter((dims[i+1], dims[i]), self.weight_scale) for i in range(self.n)]
            # l = self.setSDPlambdas()
            self.lambdas = [baseParameter(lambda_mean, lambda_std, (dims[i+1])) for i in range(self.n-1)]

            # self.initWeights()
            Q = self.calc_Q(replace_pars=False)
            while self.pretrain_epochs < 1 and not isPD(Q) and not self.loading:
                self.weight_scale *= fac
                print("reinitialising weights...")
                self.means = [weightParameter((dims[i+1], dims[i]), self.weight_scale) for i in range(self.n)]
                # l = self.setSDPlambdas()
                self.lambdas = [baseParameter(lambda_mean, lambda_std, (dims[i+1])) for i in range(self.n-1)]
                # self.initWeights()
                Q = self.calc_Q(replace_pars=False)
            
            if self.pretrain_epochs < 1 and self.replace_parameters:
                ldiags = [torch.diag(lam) for lam in self.lambdas]
                self.means = [torch.nn.Parameter(torch.mm(ldiags[i], self.means[i])) for i in range(len(self.lambdas))] + [self.means[-1]]

            # self.initWeights()
            # Q = self.calc_Q()
            # if self.pretrain_epochs < 1 and not isPD(Q):
            #     print("wtf")
            #     raise KeyboardInterrupt

        else:
            self.means = [weightParameter((dims[i+1], dims[i]), self.weight_scale) for i in range(self.n)]
        self.means += [weightParameter((dims[i+1], 1), self.weight_scale) for i in range(self.n)]

        parlist = self.means
        if mode == 'bayesian' or mode == 'bayesian_lipschitz':
            self.kl_weight = kl_fac/batch_size
            self.noise = noise
            self.sample_size = sample_size
            self.rho_mean = softplusinv(std_scale)
            self.rhos = [rhoParameter((dims[i+1], dims[i]), self.rho_mean) for i in range(self.n)]
            self.rhos += [rhoParameter((dims[i+1], 1), self.rho_mean) for i in range(self.n)]
            parlist = parlist + self.rhos
        if mode == 'lipschitz' or mode == 'bayesian_lipschitz':
            parlist = parlist + self.lambdas
        self.parameterList = nn.ParameterList(parlist)

    def initWeights(self):
        if self.mode == 'bayesian' or self.mode == 'bayesian_lipschitz':
            self.weights = [sampleParameter(self.means[i], softplus(self.rhos[i]), self.means[i].size()) for i in range(self.n*2)]
        else:
            self.weights = self.means

    def setMeans(self):
        if self.mode == 'bayesian' or self.mode == 'bayesian_lipschitz':
            self.weights = self.means
        else:
            print("This function makes sense for bayesian only!")

    def getWeights(self, with_n=False):
        self.initWeights()
        if with_n:
            return [weight.detach().numpy() for weight in self.weights], self.n
        else:
            return [weight.detach().numpy() for weight in self.weights]

    def getWeightsOnly(self, as_tensors=False, sample=False):
        if not sample and (self.mode == 'bayesian' or self.mode == 'bayesian_lipschitz'):
            self.weights = self.means
        else:
            self.initWeights()
        if as_tensors:
            return [weight for weight in self.weights[:self.n]]
        else:
            return [weight.detach().numpy() for weight in self.weights[:self.n]]

    def getWeightsVector(self):
        weights = self.getWeightsOnly(as_tensors=True)
        wvec = weights[0].flatten()
        for i in range(1, len(weights)):
            wvec = torch.hstack((wvec, weights[i].flatten()))
        return wvec

    def forward(self, x, skip_last=False):
        if len(x.shape) < 2:
            x = x.reshape(-1, 1)
        res = x
        res = res.transpose(0, -1)
        if (self.mode == "lipschitz" or self.mode == 'bayesian_lipschitz') and self.replace_parameters:
            ldiags = [torch.diag(lam) for lam in self.lambdas]
            weights = [torch.mm(torch.inverse(ldiags[i]), self.weights[i]) for i in range(len(ldiags))]
        else:
            weights = self.weights
        for i in range(self.n-1):
            res = self.act(torch.mm(weights[i], res) + self.weights[i+self.n])
        if skip_last:
            return res
        else:
            return self.act_out((torch.mm(self.weights[self.n-1], res) + self.weights[-1]).transpose(0, -1))

    def setSDPlambdas(self, replace_pars=None):
        if replace_pars == None:
            replace_pars = self.replace_parameters
        dims = self.dims
        means = self.getWeightsOnly(as_tensors=True)
        if replace_pars and self.lambdas is not None:
            ldiags = [torch.diag(lam) for lam in self.lambdas]
            means = [torch.mm(torch.inverse(ldiags[i]), means[i]) for i in range(len(ldiags))] + [means[-1]]
        l, lambdas = solveLipSDP([mean.detach().numpy() for mean in means])
        self.lambdas = [torch.nn.Parameter(torch.tensor(lambdas[sum(dims[1:i]):sum(dims[1:i+1])])) for i in range(1, self.n)]
        return l

    def calc_Q(self, replace_pars=None):
        if replace_pars == None:
            replace_pars = self.replace_parameters
        ldiags = [torch.diag(lam) for lam in self.lambdas]
        Q = torch.tensor(np.zeros((sum(self.dims), sum(self.dims))), dtype=torch.float32)

        Q[0:self.dims[0], 0:self.dims[0]] = self.l2*torch.eye(self.dims[0])

        for i in range(self.n-1):
            linds = np.arange(sum(self.dims[0:i]), sum(self.dims[0:i+1]))
            cinds = np.arange(sum(self.dims[0:i+1]), sum(self.dims[0:i+2]))
            if not replace_pars:
                Q[np.ix_(linds, cinds)] = -torch.mm(torch.transpose(self.means[i], 0, 1), ldiags[i])
                Q[np.ix_(cinds, linds)] = -torch.mm(ldiags[i], self.means[i])
            else:
                Q[np.ix_(linds, cinds)] = -torch.transpose(self.means[i], 0, 1)
                Q[np.ix_(cinds, linds)] = -self.means[i]
            Q[np.ix_(cinds, cinds)] = 2*ldiags[i]

        i += 1
        linds = np.arange(sum(self.dims[0:i]), sum(self.dims[0:i+1]))
        cinds = np.arange(sum(self.dims[0:i+1]), sum(self.dims[0:i+2]))

        Q[np.ix_(linds, cinds)] = -torch.transpose(self.means[self.n-1], 0, 1)
        Q[np.ix_(cinds, linds)] = -self.means[self.n-1]
        Q[np.ix_(cinds, cinds)] = torch.eye(self.dims[-1])

        return Q

    def getState(self):
        state = {
            'state_dict': self.state_dict()
        }

        for attribute, value in self.__dict__.items():
            state[attribute] = value

        return state

    def restoreState(self, state):
        dims = state['dims']
        # self = NetModel(dims)
        self.load_state_dict(state['state_dict'])

        for attribute, value in state.items():
            if attribute != "state_dict":
                setattr(self, attribute, value)

        return self