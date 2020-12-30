import numpy as np
import matplotlib.pyplot as plt
from pickle import load


############# PLOT ##############

def cauchy_selection_loss_numpy(y_true, y_predicted):
    y_max = 1
    y_min = -1
    gamma = 1
    r = (y_true - y_predicted)/gamma
    epsilon = 10**-6

    tail_term = np.log(1 + np.square(r))
    selection_term = np.log(np.arctan((y_max - y_predicted)/gamma) - np.arctan((y_min - y_predicted)/gamma) + epsilon)

    loss = tail_term + selection_term
    return np.mean(loss, axis=-1)

def loss(y_true, y_predicted, scaler):
    y_predicted1 = scaler.transform(y_predicted.reshape(-1, 1)).flatten()
    y_true1 = scaler.transform(y_true.reshape(-1, 1)).flatten()
    return cauchy_selection_loss_numpy(y_true1, y_predicted1)


p_tr10 = np.load("predicted_training_10.npy")
t_tr10 = np.load("true_training_10.npy")

p10 = np.load("predicted_val_10.npy")
t10 = np.load("true_val_10.npy")

p_tr20 = np.load("predicted_training_20.npy")
t_tr20 = np.load("true_training_20.npy")

p20 = np.load("predicted_val_20.npy")
t20 = np.load("true_val_20.npy")

p_tr35 = np.load("predicted_training_35.npy")
t_tr35 = np.load("true_training_35.npy")

p35 = np.load("predicted_val_35.npy")
t35 = np.load("true_val_35.npy")

p_tr50 = np.load("predicted_training_50.npy")
t_tr50 = np.load("true_training_50.npy")

p50 = np.load("predicted_val_50.npy")
t50 = np.load("true_val_50.npy")

p_tr100 = np.load("predicted_training_100.npy")
t_tr100 = np.load("true_training_100.npy")

p100 = np.load("predicted_val_100.npy")
t100 = np.load("true_val_100.npy")

scaler_training_set = load(open('../mse/scaler_output.pkl', 'rb'))

f, axes = plt.subplots(4, 2, figsize=(12, 7), sharey=True, sharex=True)

axes[0,0].scatter(t_tr10, p_tr10, s=0.1, label="Epoch 10, L=%.3f" % loss(t_tr10, p_tr10, scaler_training_set))
axes[0, 1].scatter(t10, p10, s=0.2, label="Epoch 10, L=%.3f" % loss(t10, p10, scaler_training_set))
axes[0,0].set_title("Training set")
axes[0,1].set_title("Validation set")

axes[1,0].scatter(t_tr20, p_tr20, s=0.1, label="Epoch 20, L=%.3f" % loss(t_tr20, p_tr20, scaler_training_set))
axes[1, 1].scatter(t20, p20, s=0.2, label="Epoch 20, L=%.3f" % loss(t20, p20, scaler_training_set))

axes[2,0].scatter(t_tr35, p_tr35, s=0.1, label="Epoch 35, L=%.3f" % loss(t_tr35, p_tr35, scaler_training_set))
axes[2,1].scatter(t35, p35, s=0.2, label="Epoch 35, L=%.3f" % loss(t35, p35, scaler_training_set))

axes[3,0].scatter(t_tr50, p_tr50, s=0.1, label="Epoch 50, L=%.3f" % loss(t_tr50, p_tr50, scaler_training_set))
axes[3,1].scatter(t50, p50, s=0.2, label="Epoch 50, L=%.3f" % loss(t50, p50, scaler_training_set))

# axes[3,0].scatter(t_tr100, p_tr100, s=0.1, label="Epoch 100, MSE=%.3f" % mse(t_tr50, p_tr50))
# axes[3,1].scatter(t100, p100, s=0.2, label="Epoch 100, MSE=%.3f" % mse(t50, p50))

for ax in axes.flatten():
    ax.plot([t_tr20.min(), t_tr20.max()], [t_tr20.min(), t_tr20.max()], color="grey")
    ax.legend(loc=2, fontsize=13)

plt.subplots_adjust(left=0.08, top=0.94, bottom=0.12, wspace=0, hspace=0)
axes[3,1].set_ylim(9.5, 14.8)
f.text(0.5, 0.01,r"$\log(M_{\mathrm{truth}}/M_\odot)$")
f.text(0.01, 0.4,r"$\log(M_{\mathrm{predicted}}/M_\odot)$", rotation=90)




# f, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
# plt.subplots_adjust(wspace=0, top=0.92, left=0.1)
# truth_values = [-1, 0, 1]
# xs = [np.linspace(-1.5, -0.5, 10000), np.linspace(-0.5, 0.5, 100000), np.linspace(0.5, 1.5, 100000)]
# for i in range(3):
#     axes[i].plot(xs[i], squared_error_numpy(truth_values[i], xs[i]) -
#                  squared_error_numpy(truth_values[i], truth_values[i]), label=r"$\mathcal{L}_\mathrm{MSE}$")
#     axes[i].plot(xs[i], L.loss_range(truth_values[i], xs[i]) -
#                  L.loss_range(truth_values[i], truth_values[i]), label=r"$\mathcal{L}_C$")
#     axes[i].plot(xs[i], L.loss(truth_values[i], xs[i]) -
#                  L.loss(truth_values[i], truth_values[i]), label=r"$\mathcal{L}_B$")
#     axes[i].axvline(x=truth_values[i], ls="--", color="grey")
#     axes[i].set_xlabel("x")
#
# axes[1].legend(loc="best")
# plt.ylim(-0.1, 5)
# axes[0].set_ylabel("Loss")


tr_bound = np.loadtxt("/Users/lls/Desktop/cauchy_selec_bound/test/training.log", delimiter=",", skiprows=1)
tr_gamma = np.loadtxt("/Users/lls/Desktop/cauchy_selec_gamma_bound/training.log", delimiter=",", skiprows=1)

tr_bound = tr
f, axes = plt.subplots(1, 3, sharex=True, figsize=(12, 5))
f, axes = plt.subplots(1, 1)

axes.plot(tr_bound[:,0], tr_bound[:, 1], color="C0", label="$\gamma = 0.2$")
axes.plot(tr_gamma[:,0], tr_gamma[:, 1], color="C1", label="$\gamma$ trainable")
axes.plot(tr_bound[:,0], tr_bound[:, 4], color="C0", ls="--")
axes.plot(tr_gamma[:,0], tr_gamma[:, 5], color="C1", ls="--")
axes.set_ylabel("Loss", fontsize=14)

# axes[1].plot(tr_bound[:,0], tr_bound[:, 2], color="C0")
# # axes[1].plot(tr_gamma[:,0], tr_gamma[:, 3], color="C1")
# axes[1].plot(tr_bound[:,0], tr_bound[:, 5], color="C0", ls="--")
# # axes[1].plot(tr_gamma[:,0], tr_gamma[:, 7], color="C1", ls="--")
# axes[1].set_ylabel("MAE", fontsize=14)
#
# axes[2].plot(tr_bound[:,0], tr_bound[:, 3], color="C0")
# # axes[2].plot(tr_gamma[:,0], tr_gamma[:, 4], color="C1")
# axes[2].plot(tr_bound[:,0], tr_bound[:, 6], color="C0", ls="--")
# # axes[2].plot(tr_gamma[:,0], tr_gamma[:, 8], color="C1", ls="--")
# axes[2].set_ylabel("MSE", fontsize=14)
# axes[1].set_xlabel("Epoch", fontsize=14)

for ax in axes:
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

plt.subplots_adjust(left=0.07, bottom=0.14, wspace=0.35, right=0.96, top=0.91)



############## PLOT #################

scaler_training_set = load(open('../mse/scaler_output.pkl', 'rb'))
epochs = ["15", "25"]

f, axes = plt.subplots(2, len(epochs), figsize=(13, 7), sharey=True, sharex=True)

for i, epoch in enumerate(epochs):
    p = np.load("predicted_training_" + epoch + ".npy")
    t = np.load("true_training_" + epoch + ".npy")
    if i == 0:
        axes[0, i].scatter(t, p, s=0.01, color="C0", label="Training set, L=%.3f" % loss(t, p, scaler_training_set))
    else:
        axes[0, i].scatter(t, p, s=0.01, color="C0", label="L=%.3f" % loss(t, p, scaler_training_set))
    axes[0, i].set_title("Epoch " + epoch, fontsize=13)

    p = np.load("predicted_val_" + epoch + ".npy")
    t = np.load("true_val_" + epoch + ".npy")
    if i == 0:
        axes[1, i].scatter(t, p, s=0.1, color="C1", label="Validation set, L=%.3f" % loss(t, p, scaler_training_set))
    else:
        axes[1, i].scatter(t, p, s=0.1, color="C1", label="L=%.3f" % loss(t, p, scaler_training_set))

for ax in axes.flatten():
    ax.plot([t.min(), t.max()], [t.min(), t.max()], color="grey")
    ax.legend(loc=2, fontsize=13)

plt.subplots_adjust(left=0.08, top=0.94, bottom=0.12, wspace=0, hspace=0)
axes[0,0].set_ylim(10.3, 13.1)
# f.text(0.5, 0.01,r"$\log(M_{\mathrm{truth}}/M_\odot)$")
axes[1, 1].set_xlabel(r"$\log(M_{\mathrm{truth}}/M_\odot)$", fontsize=16)
f.text(0.01, 0.4,r"$\log(M_{\mathrm{predicted}}/M_\odot)$", rotation=90, fontsize=16)




def plot_diff_predicted_true_mass_ranges(predictions, truths, mass_bins, xbins, figure=None,
                                           figsize=(10, 5.1),
                                           col_truth="dimgrey", lw=1.8,
                                          density=True):
    if figure is None:
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True, sharex=True)
    else:
        f, (ax1, ax2, ax3) = figure[0], figure[1]

    ax1.axvline(x=0, color=col_truth, ls="--")
    ax2.axvline(x=0, color=col_truth, ls="--")
    ax3.axvline(x=0, color=col_truth, ls="--")

    pred_low = (truths >= mass_bins[0]) & (truths < mass_bins[1])
    pred_mid = (truths >= mass_bins[1]) & (truths < mass_bins[2])
    pred_high = (truths >= mass_bins[2]) & (truths < mass_bins[3])

    _ = ax1.hist(predictions[pred_low] - truths[pred_low], bins=xbins,
                 histtype="step", density=density, lw=lw, color="C0")
    ax1.set_title(r"$ %.2f \leq \log(M_{\mathrm{true}}) \leq %.2f$" % (mass_bins[0], mass_bins[1]))

    _ = ax2.hist(predictions[pred_mid] - truths[pred_mid], bins=xbins,
                 histtype="step", density=density, lw=lw, color="C0")
    ax2.set_title(r"$%.2f \leq \log(M_{\mathrm{true}}) \leq %.2f$" % (mass_bins[1], mass_bins[2]))

    dd = ax3.hist(predictions[pred_high] - truths[pred_high], bins=xbins,
                 histtype="step", density=density, lw=lw, color="C0")
    ax3.set_title(r"$%.2f \leq \log(M_{\mathrm{true}}) \leq %.2f$" % (mass_bins[2], mass_bins[3]))

    plt.subplots_adjust(wspace=0, bottom=0.14, left=0.08)

    ax1.set_ylabel(r"$n_{\mathrm{particles}}$")
    ax1.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax2.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax3.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax3.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    return f, (ax1, ax2, ax3)