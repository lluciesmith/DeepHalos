import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_rms(a, n=3) :
    i = 0
    j = n
    rms = []
    while j <= len(a):
        a_i = a[i:j]
        rms.append(np.std(a_i))
        i += 1
        j += 1
    return np.array(rms)


def plot_moving_average_and_err(auc_mix, auc4, auc5, n=10, title=True):

    final_mix, mix_std = moving_average(auc_mix, n=n)[-1], moving_rms(auc_mix, n=n)[-1]
    final_auc4, auc4_std = moving_average(auc4, n=n)[-1], moving_rms(auc4, n=n)[-1]
    final_auc5, auc5_std = moving_average(auc5, n=n)[-1], moving_rms(auc5, n=n)[-1]

    plt.errorbar(moving_average(np.arange(len(auc_mix)), n=n), moving_average(auc_mix, n=n),
                 yerr=moving_rms(auc_mix, n=n), label=r"mixed sims (%.3f $\pm$ %.3f) " % (final_mix, mix_std))
    plt.errorbar(moving_average(np.arange(len(auc5)), n=n), moving_average(auc5, n=n), yerr=moving_rms(auc5, n=n),
                 label=r"sequential 5 sims (%.3f $\pm$ %.3f) " % (final_auc5, auc5_std))
    plt.errorbar(moving_average(np.arange(len(auc4)), n=n), moving_average(auc4, n=n), yerr=moving_rms(auc4, n=n),
                 label=r"sequential 4 sims (%.3f $\pm$ %.3f) " % (final_auc4, auc4_std))

    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    if title is True:
        plt.title("Moving average and std (size window = %i)" % n)
        plt.subplots_adjust(bottom=0.15, top=0.92)
    else:
        plt.subplots_adjust(bottom=0.15, top=0.92)
    plt.savefig("/Users/lls/Documents/deep_halos_files/binary/auc_comparison_new.pdf")

h = np.load("histories.npy", allow_pickle=True)
epochs = np.arange(111)
num_loops = len(epochs)/3

h_all = {}
for k in h[0].keys():
    h_all[k] = [di[k] for di in h]
    h_all[k] = list(np.array(h_all[k]).flatten())

f, axes = plt.subplots(nrows=1, ncols=2, sharex="all", figsize=(12, 5.5))
ax0 = axes[0]
ax0.plot(epochs, h_all['auc_train_i'], label="training", color="k")
ax0.plot(epochs, h_all['auc_val_1'], label="val - sim1")
# ax0.plot(epochs, h_all['auc_val_2'], label="val - sim2")
ax0.legend(loc="best")
ax0.set_xlabel("Epochs")
ax0.set_ylabel("AUC")
ax0.set_title("AUC")

ax1 = axes[1]
ax1.plot(epochs, h_all['loss'], color="k")
ax1.plot(epochs, h_all['val_loss'])
#ax1.plot(epochs, h_all['loss_val_2'])
ax1.set_ylabel("Binary cross-entropy (loss)")
ax1.set_xlabel("Epochs")
ax1.set_title("LOSS")
plt.subplots_adjust(left=0.08, top=0.9, wspace=0.3, bottom=0.14)

colors = ["b", "orange", "r", "m", "brown"]
sims = ["0", "2", "3", "4", "5"]
for i in range(5):
# colors = ["b", "r", "m", "brown"]
# sims = ["0", "3", "4", "5"]
# for i in range(4):
    for j in range(15):
        if j == 0:
            ax0.scatter(epochs[(3 * i) + 15 * j:(3 * i + 3) + 15 * j],
                        h_all['auc_train_i'][(3 * i) + 15 * j:(3 * i + 3) + 15 * j],
                        color=colors[i], s=20)
            ax1.scatter(epochs[(3 * i) + 15 * j:(3 * i + 3) + 15 * j],
                        h_all['loss'][(3 * i) + 15 * j:(3 * i + 3) + 15 * j],
                        color=colors[i], s=20, label='train - sim' + str(sims[i]))
        else:
            ax0.scatter(epochs[(3 * i) + 15 * j:(3 * i + 3) + 15 * j],
                        h_all['auc_train_i'][(3 * i) + 15 * j:(3 * i + 3) + 15 * j],
                        color=colors[i], s=20)
            ax1.scatter(epochs[(3 * i) + 15 * j:(3 * i + 3) + 15 * j],
                        h_all['loss'][(3 * i) + 15 * j:(3 * i + 3) + 15 * j],
                        color=colors[i], s=20)

ax1.legend(loc="best")
# plt.savefig("bug_fix_training_set_sim0.png")


colors = ["b", "r", "m", "brown"]
sims = ["0", "3", "4", "5"]
for i in range(4):
    for j in range(15):
        if j == 0:
            plt.scatter(epochs[(3 * i) + 12 * j:(3 * i + 3) + 12 * j],
                        h_all['mean_absolute_error'][(3 * i) + 12 * j:(3 * i + 3) + 12 * j],
                        color=colors[i], s=20)
        else:
            plt.scatter(epochs[(3 * i) + 12 * j:(3 * i + 3) + 12 * j],
                        h_all['mean_absolute_error'][(3 * i) + 12 * j:(3 * i + 3) + 12 * j],
                        color=colors[i], s=20)
