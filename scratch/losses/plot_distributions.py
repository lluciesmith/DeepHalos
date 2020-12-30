import numpy as np
import matplotlib.pyplot as plt
from pickle import load
import scipy.stats
import sys
sys.path.append("/Users/lls/Documents/Projects/DeepHalos")
import dlhalos_code.loss_functions as lf

# path = '/Users/lls/Documents/deep_halos_files/regression/test_lowmass/reg_10000_perbin/larger_net' \
#        '/cauchy_selec_gamma_bound/model_every_epoch/'
#
# p = np.load(path + "predicted_larger_val_19.npy")
# t = np.load(path + "true_larger_val_19.npy")
# scaler_training_set = load(open(path + '../scaler_output.pkl', 'rb'))
# gamma=0.22452055

path = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4/'
p = np.load(path + "predicted_sim_6_epoch_38.npy")
t = np.load(path + "true_sim_6_epoch_38.npy")
scaler_training_set = load(open(path + 'scaler_output.pkl', 'rb'))
gamma = 0.38804

p_rescaled = scaler_training_set.transform(p.reshape(-1, 1)).flatten()
t_rescaled = scaler_training_set.transform(t.reshape(-1, 1)).flatten()


slice1 = (-0.91, -0.89)
slice2 = (-0.51, -0.49)
slice3 = (-0.01, 0.01)
slice4 = (0.49, 0.51)
slice5 = (0.89, 0.91)
slice6 = (0.94, 0.96)
slices = [slice1, slice2, slice3, slice4, slice5, slice6]


L = lf.ConditionalCauchySelectionLoss(gamma=gamma)
bins = np.linspace(-1, 1, 60, endpoint=True)

f, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 8))
axes2 = axes.flatten()

for i, slice in enumerate(slices):
    ax = axes2[i]

    ind = np.where((p_rescaled >= slice[0]) * (p_rescaled <= slice[1]))[0]
    mean_x = np.mean([slice[0], slice[1]])

    _ = ax.hist(t_rescaled[ind], bins=bins, histtype="step", lw=2, label="$%.2f \leq x \leq %.2f $" % slice,
                density=True)
    # __ = ax.plot(bins, scipy.stats.cauchy.pdf(bins, mean_x, 0.22452055))

    y_pred = np.repeat(mean_x, len(bins))
    liklih = np.exp(- L.loss(bins.reshape(-1, 1), y_pred.reshape(-1, 1)))
    ___ = ax.plot(bins, liklih, color="k")

    ax.axvline(x=np.mean([slice[0], slice[1]]), color="k")

    if i > 1:
        ax.legend(loc=2)
    else:
        ax.legend(loc="best")
    # ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

plt.subplots_adjust(left=0.02, bottom=0.12, wspace=0, hspace=0)
f.text(0.5, 0.01,r"$d$")


