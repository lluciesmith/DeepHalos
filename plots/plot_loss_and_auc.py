import numpy as np
import matplotlib.pyplot as plt

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