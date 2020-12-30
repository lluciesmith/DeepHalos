import numpy as np
import matplotlib.pyplot as plt
from plots import plots_for_predictions as pp

path = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/'
p_raw = np.load(path + "lr5e-5/seed_20/predicted_sim_6_epoch_09.npy")
t_raw = np.load(path + "lr5e-5/seed_20/true_sim_6_epoch_09.npy")
p_av = np.load(path + "averaged_boxes/log_alpha_-4.3/predicted_sim_6_epoch_32.npy")
t_av = np.load(path + "averaged_boxes/log_alpha_-4.3/true_sim_6_epoch_32.npy")
np.allclose(t_av, t_raw)

mass_bins = np.linspace(11, t_av.max(), 17, endpoint=True)
bins = np.linspace(10, 14, 40)

f1 = pp.plot_predicted_in_fine_mass_bins(p_av, t_av, p_raw, t_raw, mass_bins, bins, label1="averaged-density",
                                         label2="raw-density", figsize=(11.69, 12))
plt.savefig("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4.3/"
            "comparison_averaged_raw_fine_mass_bins.pdf")
plt.clf()

t_GBT, p_GBT = pp.get_truth_predictions_GBT()
f2 = pp.plot_predicted_in_fine_mass_bins(p_av, t_av, p_GBT, t_GBT, mass_bins, bins, label1="averaged-density",
                                         label2="GBT", figsize=(11.69, 12))
plt.savefig("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4.3/"
            "comparison_averaged_GBT_fine_mass_bins.pdf")