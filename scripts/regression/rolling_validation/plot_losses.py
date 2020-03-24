import numpy as np
import matplotlib.pyplot as plt


paths = '/Users/lls/Documents/deep_halos_files/regression/rolling_val/'
paths_all = [paths + '0dropout/', paths + '0.1dropout/', paths + '0.2dropout/', paths + '0.3dropout/',
             paths + 'no_sim3/']
dropout_pc = ["0", "10", "20", "30", "40"]

f, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 7))
axes = np.concatenate(axes)
n = 5
for i in range(6):
    ax = axes[i]
    if i == 5:
        ax.plot([], [], label="training", color="C0")
        ax.plot([], [], label="validation", color="C1")

        ax.plot([], [], label="training (end of epoch + no dropout)",
                color="k")
        ax.plot([], [], color="C2",
                label="sim-7 (not used during training)")

        ax.plot([], [], label="all particles", color="dimgrey", ls="-", lw=2)
        ax.plot([], [], label="particles with $\log(M) \leq 13.5$", color="dimgrey", ls="--", lw=2)

        ax.legend(loc="center", fontsize=14)
        ax.axis('off')
    else:
        if i == 0:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.01, 0.58)
        path = paths_all[i]

        tr = np.loadtxt(path + "training.log", delimiter=",", skiprows=1)
        loss_training_sim7 = np.load(path + "loss_training_and_sim7.npy")

        ax.plot(tr[n:, 0], tr[n:, 1], label="training", color="C0")
        ax.plot(tr[n:, 0], tr[n:, 2], label="validation", color="C1")

        ax.plot(loss_training_sim7[:, 0], loss_training_sim7[:, 1], label="training (end of epoch + no dropout)",
                color="k")
        ax.scatter(loss_training_sim7[:,0], loss_training_sim7[:,1], color="k", s=5)
        ax.plot(loss_training_sim7[:, 0], loss_training_sim7[:, 2], color="C2",
                label="sim-7 (not used during training)")
        ax.scatter(loss_training_sim7[:,0], loss_training_sim7[:,2], color="C2", s=5)

        if i == 2:
            ax.set_ylabel("Loss")
        if i == 4:
            ax.set_xlabel("Epoch")

        if i in [2, 3, 4]:
            ax.text(0.7, 0.9, dropout_pc[i] + r"$\%$ dropout", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.7, 0.7, dropout_pc[i] + r"$\%$ dropout", ha='center', va='center', transform=ax.transAxes)

for i in range(6):
    ax = axes[i]
    if i == 5:
        pass
    else:
        if i == 0:
            ax.set_yscale("log")
        elif i in [3, 4]:
            ax.set_ylim(0.12, 0.5)
        else:
            ax.set_ylim(0.01, 0.58)
        path = paths_all[i]

        tr = np.loadtxt(path + "training.log", delimiter=",", skiprows=1)
        # loss_training_sim7 = np.load(path + "loss_training_and_sim7.npy")
        loss_training_sim7_above_135 = np.load(path + "loss_training_sim7_validaton_above_135Msol.npy")

        ax.plot(loss_training_sim7_above_135[:, 0], loss_training_sim7_above_135[:, 3], label="validation", color="C1", ls="--")

        ax.plot(loss_training_sim7_above_135[:, 0], loss_training_sim7_above_135[:, 1], label="training (end of epoch + no dropout)",
                color="k", ls="--")
        ax.scatter(loss_training_sim7_above_135[:,0], loss_training_sim7_above_135[:,1], color="k", s=5)
        ax.plot(loss_training_sim7_above_135[:, 0], loss_training_sim7_above_135[:, 2], color="C2",
                label="sim-7 (not used during training)", ls="--")
        ax.scatter(loss_training_sim7_above_135[:,0], loss_training_sim7_above_135[:,2], color="C2", s=5)

plt.subplots_adjust(wspace=0.15, hspace=0.05)
plt.savefig(paths + '')
