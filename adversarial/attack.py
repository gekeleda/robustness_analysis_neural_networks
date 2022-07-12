import numpy as np
from scipy.stats import describe, gaussian_kde
import torch
from torch import nn
from train_sine import SineDataset, LitProgressBar
from matplotlib import pyplot as plt
from matplotlib import animation
from net import *
from tqdm import tqdm

def noise_example(sample_x, eps=1/255.): # adds random noise to a sample
        return torch.normal(sample_x, eps*torch.ones_like(sample_x))

def adversarial_example(sample_x, sample_y, net, eps=1/255.): # transforms example data into adversarial example using fgsm
    if not sample_x.requires_grad:
        sample_x = sample_x.clone().detach().requires_grad_(True)

    lossf = nn.CrossEntropyLoss() # cross entropy loss
    pred = net.forward(sample_x)
    loss = lossf(pred, sample_y)
    sample_x.retain_grad()
    loss.backward()
    grad = sample_x.grad
    adversarial_x = sample_x + eps * torch.sign(grad)
    adversarial_x = torch.max(adversarial_x, torch.zeros_like(adversarial_x)) # clip negative values to 0
    adversarial_x = torch.min(adversarial_x, torch.ones_like(adversarial_x)) # clip >1 values to 1
    return adversarial_x

def loop_through_adv_examples(net, eps=0.035, show_same=False): # show different adversarial attacks for samples from test data
    iloader = iter(net.test_loader)
    samples_x, samples_y = next(iloader)
    for index in range(len(samples_x)):
        sample_x, sample_y = samples_x[index].float().reshape(1,-1).clone().detach().requires_grad_(True), samples_y[index].reshape(1, -1)
        sample_pred = net.forward(sample_x)
        adv_x = adversarial_example(sample_x, sample_y, net, eps=eps)
        adv_y = net.forward(adv_x)

        pert_x = adv_x - sample_x

        sample_pred_lab = label(sample_pred).item()
        adv_y_lab = label(adv_y).item()

        if not show_same and adv_y_lab == sample_pred_lab:
            continue

        fig = plt.figure()
        # fig.suptitle(index)
        plt.subplot(1,3,1)
        plt.imshow(sample_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none')
        plt.title("Sample prediction: {}".format(sample_pred_lab) + "\n confidence: " + str(round(sample_pred[0][sample_pred_lab].item(), 3)), c='red')
        plt.xticks([])
        plt.yticks([])
        plt.clim(0., 1.)
        plt.subplot(1,3,2)
        plt.imshow(pert_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none')
        plt.title("Perturbation", c='red')
        plt.xticks([])
        plt.yticks([])
        plt.clim(0., 1.)
        plt.subplot(1,3,3)
        plt.imshow(adv_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none')
        plt.title("Adversarial prediction: {}".format(adv_y_lab) + "\n confidence: " + str(round(adv_y[0][adv_y_lab].item(), 3)), c='red')
        plt.xticks([])
        plt.yticks([])
        plt.clim(0., 1.)
        plt.tight_layout()
        plt.show()

def animate_adv_examples(net, eps_range=(0., 1.)): # animate adv examples with increasing strength
    iloader = iter(net.test_loader)
    samples_x, samples_y = next(iloader)
    for index in range(len(samples_x)):
        sample_x, sample_y = samples_x[index].float().reshape(1,-1).clone().detach().requires_grad_(True), samples_y[index].reshape(1, -1)
        sample_pred = net.forward(sample_x)
        sample_pred_lab = label(sample_pred).item()

        eps_min, eps_max = eps_range
        eps = eps_min

        adv_x = adversarial_example(sample_x, sample_y, net, eps=eps)
        adv_y = net.forward(adv_x)
        pert_x = adv_x - sample_x
        adv_y_lab = label(adv_y).item()

        fig = plt.figure()
        fig.suptitle(index)
        plt.subplot(1,3,1)
        plt.imshow(sample_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none')
        plt.title("Sample prediction: {}".format(sample_pred_lab) + "\n confidence: " + str(round(sample_pred[0][sample_pred_lab].item(), 3)), c='red')
        plt.xticks([])
        plt.yticks([])
        plt.clim(0., 1.)
        plt.subplot(1,3,2)
        pert_img = plt.imshow(pert_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none')
        plt.title("Perturbation", c='red')
        plt.xticks([])
        plt.yticks([])
        plt.clim(0., 1.)
        plt.subplot(1,3,3)
        adv_img = plt.imshow(adv_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none')
        adv_txt = plt.title("Adversarial prediction: {}".format(adv_y_lab) + "\n confidence: " + str(round(adv_y[0][adv_y_lab].item(), 3)), c='red')
        plt.xticks([])
        plt.yticks([])
        plt.clim(0., 1.)
        plt.tight_layout()

        def init():
            return [pert_img, adv_img, adv_txt]
        def animate(i, *fargs):
            eps_list = fargs[0]
            eps_cur = eps_list[i]
            adv_x = adversarial_example(sample_x, sample_y, net, eps=eps_cur)
            adv_y = net.forward(adv_x)
            pert_x = adv_x - sample_x
            adv_y_lab = label(adv_y).item()
            pert_img.set(array=pert_x.detach().numpy().reshape((14,14)))
            adv_img.set(array=adv_x.detach().numpy().reshape((14,14)))
            adv_txt.set_text("Adversarial prediction: {}".format(adv_y_lab) + "\n confidence: " + str(round(adv_y[0][adv_y_lab].item(), 3)))
            return [pert_img, adv_img, adv_txt]

        frames = 200
        # step = (eps_max - eps_min) / frames
        eps_list = np.linspace(eps_min, eps_max, frames)
 
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, fargs=(eps_list,), init_func=init,
                               frames=frames, interval=20, blit=False)
        
        # anim.save('test_anim.gif', fps=1000/20)

        plt.show()

def lipschitz_superiority_examples(nomnet, lipnet, eps=0.035, show_same=False): # find examples where nominal NN is fooled but LNN is not
    test_loader = lipnet.test_loader
    iloader = iter(lipnet.test_loader)
    samples_x, samples_y = next(iloader)
    for index in range(len(samples_x)):
        sample_x, sample_y = samples_x[index].float().reshape(1,-1).clone().detach().requires_grad_(True), samples_y[index].reshape(1, -1)
        lip_sample_pred = lipnet.forward(sample_x)
        lip_adv_x = adversarial_example(sample_x, sample_y, lipnet, eps=eps)
        lip_adv_y = lipnet.forward(lip_adv_x)
        nom_sample_pred = nomnet.forward(sample_x)
        nom_adv_x = adversarial_example(sample_x, sample_y, nomnet, eps=eps)
        nom_adv_y = nomnet.forward(nom_adv_x)

        lip_pert_x = lip_adv_x - sample_x
        nom_pert_x = nom_adv_x - sample_x

        sample_y_lab = label(sample_y)
        lip_sample_pred_lab = label(lip_sample_pred).item()
        lip_adv_y_lab = label(lip_adv_y).item()
        nom_sample_pred_lab = label(nom_sample_pred).item()
        nom_adv_y_lab = label(nom_adv_y).item()

        if nom_sample_pred_lab != sample_y_lab or nom_adv_y_lab == nom_sample_pred_lab or not (sample_y_lab == lip_sample_pred_lab and sample_y_lab == lip_adv_y_lab):
            continue

        fig = plt.figure()
        # fig.suptitle(index)
        sfig1, sfig2 = fig.subfigures(nrows=2, ncols=1)
        sfig1.suptitle('nominal net')
        sfig2.suptitle('lipschitz net')

        splot1, splot2, splot3 = sfig1.subplots(nrows=1, ncols=3)
        splot1.imshow(sample_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none', vmin=0., vmax=1.)
        splot1.set_title("prediction: {}".format(nom_sample_pred_lab) + "\n confidence: " + str(round(nom_sample_pred[0][nom_sample_pred_lab].item(), 3)), c='red')
        splot1.set_xticks([])
        splot1.set_yticks([])
        # splot1.set_clim(0., 1.)
        splot2.imshow(nom_pert_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none', vmin=0., vmax=1.)
        splot2.set_title("perturbation", c='red')
        splot2.set_xticks([])
        splot2.set_yticks([])
        # splot2.set_clim(0., 1.)
        splot3.imshow(nom_adv_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none', vmin=0., vmax=1.)
        splot3.set_title("adversarial prediction: {}".format(nom_adv_y_lab) + "\n confidence: " + str(round(nom_adv_y[0][nom_adv_y_lab].item(), 3)), c='red')
        splot3.set_xticks([])
        splot3.set_yticks([])
        # splot3.set_clim(0., 1.)

        splot4, splot5, splot6 = sfig2.subplots(nrows=1, ncols=3)
        splot4.imshow(sample_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none', vmin=0., vmax=1.)
        splot4.set_title("prediction: {}".format(lip_sample_pred_lab) + "\n confidence: " + str(round(lip_sample_pred[0][lip_sample_pred_lab].item(), 3)), c='red')
        splot4.set_xticks([])
        splot4.set_yticks([])
        # splot4.set_clim(0., 1.)
        splot5.imshow(lip_pert_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none', vmin=0., vmax=1.)
        splot5.set_title("perturbation", c='red')
        splot5.set_xticks([])
        splot5.set_yticks([])
        # splot5.set_clim(0., 1.)
        splot6.imshow(lip_adv_x.detach().numpy().reshape((14,14)), cmap='gray', interpolation='none', vmin=0., vmax=1.)
        splot6.set_title("adversarial prediction: {}".format(lip_adv_y_lab) + "\n confidence: " + str(round(lip_adv_y[0][lip_adv_y_lab].item(), 3)), c='red')
        splot6.set_xticks([])
        splot6.set_yticks([])
        # splot6.set_clim(0., 1.)
        # plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=0.30, right=None, top=0.67, wspace=None, hspace=0.3)
        plt.show()

def accuracy_plot(nets, eps_list=[0., 1e-3, 1e-2, 3e-2, 7e-2, 1e-1, 2e-1, 3e-1, 7e-1, 1e0], labels=None, mode="fgsm"):
    # plot accuracy over perturbation strength for different models
    test_loader = nets[0].test_loader
    iloader = iter(test_loader)
    samples_x, samples_y = next(iloader)
    nets_accuracies = []
    for net in nets:
        nets_accuracies.append([])
        for eps in tqdm(eps_list):
            nets_accuracies[-1].append(accuracy(net, samples_x, samples_y, eps, mode=mode))
    np.save("adversarial/adv_plot_data.npy" ,[nets_accuracies, eps_list, labels])
    print(nets_accuracies)
    for i in range(len(nets)):
        if labels is None:
            plt.plot(eps_list, nets_accuracies[i])
        else:
            plt.plot(eps_list, nets_accuracies[i], label=labels[i])
    plt.xscale('log')
    if labels is not None:
        plt.legend()

    plt.xlabel("perturbation strength [-]")
    plt.ylabel("test accuracy [-]")
    
    plt.show()

def accuracy(net, samples_x, samples_y, eps, mode="fgsm"): # calculate accuracy on test dataset for given net and pert. strength eps
    total = len(samples_x)
    correct = 0
    for index in range(len(samples_x)):
        if mode=='mnist':
            plt.imshow(samples_x[index].float().detach().numpy().reshape((14,14)), cmap='gray')
            # plt.title(label(samples_y[index].reshape(1,-1)).item(), y=-0.1)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            continue
        sample_x, sample_y = samples_x[index].float().reshape(1,-1).clone().detach().requires_grad_(True), samples_y[index].reshape(1, -1)
        sample_pred = net.forward(sample_x)
        adv_x = None
        if mode == "fgsm":
            adv_x = adversarial_example(sample_x, sample_y, net, eps=eps)
        elif mode == "noise":
            adv_x = noise_example(sample_x, eps=eps)
        adv_y = net.forward(adv_x)

        if label(adv_y).item() == label(sample_y).item():
            correct += 1
    return correct/total

parnum = 0
net0 = loadNet('saved_nets/norm/saved_nets1/net_mnist0.pt')
net3 = loadNet('saved_nets/norm/saved_nets1/net_mnist3.pt')
net4 = loadNet('saved_nets/norm/saved_nets1/net_mnist4.pt')
net8 = loadNet('saved_nets/norm/saved_nets1/net_mnist8.pt')
net8 = loadNet('saved_nets/norm/saved_nets1/net_mnist8.pt')
net12 = loadNet('saved_nets/norm/saved_nets1/net_mnist12.pt')

accuracy_plot([net0, net3, net4, net8, net12], labels=["nominal", "nominal + L2", "lipschitz", "bayesian", "bayesian-lipschitz"], mode="mnist")
# loop_through_adv_examples(net0, eps=0.035, show_same=True)
# animate_adv_examples(net0, eps_range=(0.0, 0.5))
# lipschitz_superiority_examples(net0, net4, eps=0.035)