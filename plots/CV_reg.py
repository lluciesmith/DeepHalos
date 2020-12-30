import numpy as np
import matplotlib.pyplot as plt

path = "/Users/lls/Documents/deep_halos_files/regression/test_lowmass/reg_10000_perbin/larger_net/" \
       "cauchy_selec_gamma_bound/group_reg_alpha/"
alpha_grid = [10**-j for j in np.arange(1, 5).astype("float64")]
dir = [str(s) for s in alpha_grid]

f, axes = plt.subplots(len(alpha_grid), 2, sharex=True, figsize=(12, 8))
first_row = axes[0]
second_row = axes[1]

color = ["C" + str(i) for i in range(len(alpha_grid))]

for i in range(len(alpha_grid)):
    ax = axes[i, 0]

    tr = np.loadtxt(path + "alpha_" + dir[i] + "/training.log", delimiter=",", skiprows=1)
    ax.plot(tr[:,0], tr[:,1], lw=1.5, color=color[i], label=r"$\alpha = $" + dir[i])
    ax.plot(tr[:, 0], tr[:, 2], ls="--", color=color[i], lw=1.5)
    ax.legend(loc="best")
    ax.set_ylabel('Loss')

    ax = axes[i, 1]
    g = np.load(path + "alpha_" + dir[i] + "/trained_loss_gamma.npy")
    g = np.insert(g, 0, 0.2)
    ax.plot(np.arange(1, len(g) + 1), g, color=color[i])
    ax.set_ylabel(r'$\gamma$')

plt.subplots_adjust(wspace=0.17, hspace=0.1, left=0.07, bottom=0.1)
f.text(0.5, 0.01, "Epoch")
axes[0, 0].set_yscale("log")
axes[1, 0].set_yscale("log")

plt.savefig(path + "loss_gamma_CV.png")
