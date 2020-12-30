import numpy as np
from plots import plots_for_predictions as pp
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from mlhalos import distinct_colours as dc
import matplotlib.pyplot as plt

c = dc.get_distinct(4)

path = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'
p1 = np.load(path + "seed_20/predicted_sim_6_epoch_09.npy")
t1 = np.load(path + "seed_20/true_sim_6_epoch_09.npy")

path_av = "/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4.3/"
p_av = np.load(path_av + "predicted_sim_6_epoch_32.npy")
t_av = np.load(path_av + "true_sim_6_epoch_32.npy")

col_raw = c[1]
col_av = c[0]

f1, a, m = pp.plot_histogram_predictions(p1, t1, radius_bins=False, particle_ids=None, errorbars=False,
                                         label="Raw density", color=col_raw)
f11, a1, m1 = pp.plot_histogram_predictions(p_av, t_av, radius_bins=False, particle_ids=None, errorbars=False, fig=f1,
                                            axes=a, color=col_av, label="Averaged density")
plt.savefig("/Users/lls/Documents/Papers/dlhalos_paper/averaged_vs_raw.pdf")


cmap = 'RdBu'