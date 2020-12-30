import numpy as np
import matplotlib.pyplot as plt

alphas = ["-3", "-3.5", "-4", "-5", "-6", "no_reg"]

TR = []
for alpha in alphas[:-1]:
    TR.append(np.loadtxt("log_alpha_" + alpha + "/training.log", delimiter=",", skiprows=1))
TR.append(np.loadtxt("no_reg/training.log", delimiter=",", skiprows=1))

f, axes = plt.subplots(2, 3,figsize=(12,6))
axes = axes.flatten()
for i, ax in enumerate(axes[:-1]):
    ax.plot(TR[i][:,0], TR[i][:,2], color="C" + str(i),label=r"$\log \alpha = $" + alphas[i])
    ax.plot(TR[i][:, 0], TR[i][:, 4], color="C" + str(i), ls="--")
    ax.legend(loc="best")

axes[-1].plot(TR[-1][:,0], TR[-1][:,1], color="C" + str(len(TR)), label="no reg")
axes[-1].plot(TR[-1][:,0], TR[-1][:,2], color="C" + str(len(TR)), ls="--")
axes[-1].legend(loc="best")

plt.subplots_adjust(bottom=0.12, left=0.08, right=0.95,top=0.98, wspace=0.25)
axes[-2].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[3].set_ylabel("Loss")
# axes[-1].plot(np.arange(16), l_tr_end_epoch - 0.003, color="k", label="training loss at end epoch")
axes[-1].legend(loc="best", fontsize=14)
plt.savefig("losses_alpha.png")


path_raw = "/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/alpha_-2.2/"
path_av = "/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4/"


true_training = np.load(path_av + "training_true_sim_epoch_38.npy")
av_p_training = np.load(path_av + "training_predicted_epoch_38.npy")
raw_p_training = np.load(path_raw + "training_predicted_epoch_12.npy")

true_val = np.load(path_raw + "true_sim_6_epoch_12.npy")
av_p_val = np.load(path_av + "predicted_sim_6_epoch_38.npy")
raw_p_val = np.load(path_raw + "predicted_sim_6_epoch_12.npy")

true_small_val = np.load(path_raw + "small_val_true_sim_6_epoch_12.npy")
av_p_small_val = np.load(path_av + "small_val_predicted_sim_6_epoch_38.npy")
raw_p_small_val = np.load(path_raw + "small_val_predicted_sim_6_epoch_12.npy")

ind = np.random.choice(np.arange(len(true_training)), 50000)


f, axes = plt.subplots(2, 3,figsize=(12,6), sharex=True, sharey=True)
axes[0, 0].scatter(true_training[ind], av_p_training[ind], s=0.01, label="training set")
axes[0, 1].scatter(true_training[ind], raw_p_training[ind],s=0.01)
axes[0, 2].scatter(raw_p_training[ind], av_p_training[ind], s=0.01)

axes[1, 0].scatter(true_val, av_p_val, s=0.01, label="validation set")
axes[1, 1].scatter(true_val, raw_p_val, s=0.01)
axes[1, 2].scatter(raw_p_val, av_p_val, s=0.01)

axes[1, 0].legend(loc="best", fontsize=12)
axes[0, 0].legend(loc="best", fontsize=12)

for ax in axes.flatten():
    ax.plot([true_training.min(), true_training.max()], [true_training.min(), true_training.max()], color="k")

fontsize=14
for i in range(2):
    axes[i, 0].set_xlabel("Truth", fontsize=fontsize)
    axes[i, 0].set_ylabel("Predicted (averaged)", fontsize=fontsize)
    axes[i, 1].set_xlabel("Truth", fontsize=fontsize, labelpad=10)
    axes[i, 1].set_ylabel("Predicted (raw)", fontsize=fontsize, labelpad=10)
    axes[i, 2].set_xlabel("Predicted (raw)", fontsize=fontsize, labelpad=10)
    axes[i, 2].set_ylabel("Predicted (averaged)", fontsize=fontsize, labelpad=10)
plt.subplots_adjust(bottom=0.14, left=0.08)
