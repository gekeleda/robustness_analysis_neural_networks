from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

# convenience script for plotting saved adversarial data

nets_accuracies, eps_list, labels = np.load("adversarial/adv_plot_data.npy", allow_pickle=True)
for i in range(len(nets_accuracies)):
    if labels is None:
        plt.plot(eps_list, nets_accuracies[i])
    else:
        plt.plot(eps_list, nets_accuracies[i], label=labels[i])
plt.xscale('log')
if labels is not None:
    plt.legend()

plt.xlabel("perturbation strength")
plt.ylabel("test accuracy")

plt.show()