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

