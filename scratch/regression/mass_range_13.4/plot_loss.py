import numpy as np
import matplotlib.pyplot as plt

path = "/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/"
alpha_grid = ["-2", "$-2.5$", "$-3$"]
dir = ["alpha-2", "alpha-2.5", "alpha-3"]

f, axes = plt.subplots(len(alpha_grid), 2, sharex=True, figsize=(12, 8))
first_row = axes[0]
second_row = axes[1]

color = ["C" + str(i) for i in range(len(alpha_grid))]

for i in range(len(alpha_grid)):
    ax = axes[i, 0]

    tr = np.loadtxt(path + dir[i] + "/training.log", delimiter=",", skiprows=1)
    ax.plot(tr[:-1,0], tr[:-1,2], lw=1.5, color=color[i], label=r"$\log \alpha = $" + alpha_grid[i])
    ax.plot(tr[:-1, 0], tr[:-1, 5], ls="--", color=color[i], lw=1.5)
    ax.legend(loc="best")
    ax.set_ylabel('Loss')

    ax = axes[i, 1]
    ax.plot(tr[:-1,0], tr[:-1,1], lw=1.5, color=color[i], label=r"$\alpha = $" + dir[i])
    ax.plot(tr[:-1, 0], tr[:-1, 4], ls="--", color=color[i], lw=1.5)
    ax.set_ylabel('Likelihood')
    ax.set_ylim(0.1, 0.5)
    print(tr[:-1, 4].min())

    # ax = axes[i, 1]
    # # g = np.loadtxt(path + dir[i] + "/gamma.txt", delimiter=",")
    # g = np.load(path + dir[i] + "/gamma.npy")
    # ax.plot(g, color=color[i])
    # ax.set_ylabel(r'$\gamma$')
    # ax.set_ylim(0.1, 0.4)

f.text(0.5, 0.01, "Epoch")
axes[0, 0].set_yscale("log")
axes[1, 0].set_yscale("log")
f.text(0.25, 0.95, "Training set of 200k particles randomly-sampled from 20 sims")
plt.subplots_adjust(bottom=0.14, left=0.08, top=0.92, wspace=0.18, hspace=0.1)

plt.savefig(path + "loss_gamma_CV.png")