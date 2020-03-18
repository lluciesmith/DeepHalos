import numpy as np
import matplotlib.pyplot as plt


path = '/Users/lls/Documents/deep_halos_files/regression/rolling_val/no_sim3/'
dropout_pc = "40"

tr = np.loadtxt(path + "training.log", delimiter=",", skiprows=0)
loss_training_sim7 = np.load(path + "loss_training_and_sim7.npy")

plt.plot(tr[:, 0], tr[:, 1], label="training")
plt.plot(tr[:, 0], tr[:, 2], label="validation")

plt.plot(loss_training_sim7[:, 0], loss_training_sim7[:, 1], label="training (end of epoch + no dropout)", color="k")
plt.scatter(loss_training_sim7[:,0], loss_training_sim7[:,1], color="k", s=3)
plt.plot(loss_training_sim7[:, 0], loss_training_sim7[:, 2], color="C2", label="sim-7 (not used during training)")
plt.scatter(loss_training_sim7[:,0], loss_training_sim7[:,2], color="C2", s=5)

plt.ylim(10**-3, 1.1)
plt.legend(loc="best", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(dropout_pc + r"$\%$ dropout")

plt.yscale("log")
plt.savefig(path + "loss.png")
