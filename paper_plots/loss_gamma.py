import numpy as np
import matplotlib.pyplot as plt


path = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/seed_20/'
tr = np.loadtxt("training.log", delimiter=",", skiprows=1)[:-6, :]
g = np.load("gamma.npy")[:-6]
iterations = (tr[:,0]+1) * 3125.
min_loss = np.where(tr[:,4] == tr[:,4].min())[0]

f, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(iterations, tr[:,2], label="training set")
ax[0].plot(iterations, tr[:,4], ls="--", color="C0", label="validation set")
ax[0].set_yscale("log")
ax[0].legend(loc="best")
ax[0].set_ylabel("Loss", labelpad=15)
ax[0].axvline(x=min_loss*3125, color="k")

ax[1].axvline(x=min_loss*3125, color="k")
ax[1].plot(np.insert(iterations, values=0, obj=0), g, color="C1")
ax[1].text(29000, 0.285, "Early\nStopping", ha='center', va='center', fontweight=1000, color="k", fontsize=16)
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel(r"$\gamma$", labelpad=10)

plt.subplots_adjust(bottom=0.14, hspace=0)
plt.savefig("/Users/lls/Documents/Papers/deep_halos/loss_gamma_vs_epoch.pdf")
