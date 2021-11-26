import numpy as np
from plots import plots_for_predictions as pp
from utilss import distinct_colours as dc
import matplotlib.pyplot as plt

c = dc.get_distinct(4)

path = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'
p1 = np.load(path + "seed_20/predicted_sim_6_epoch_09.npy")
t1 = np.load(path + "seed_20/true_sim_6_epoch_09.npy")

f, ax, m_bins = pp.predictions_GBT_mass_bins(color=c_gbt)
___ = pp.plot_histogram_predictions(p1, t1, radius_bins=False, errorbars=False, label="Deep learning", mass_bins=m_bins,
                                    fig=f, axes=ax, color=c_dl)
plt.subplots_adjust(left=0.05)
plt.savefig("/Users/lls/Documents/Papers/deep_halos/CNN_GBT.pdf")

ids = np.load("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/ids_larger_validation_set.npy")
f1, a, m = pp.plot_histogram_predictions(p1, t1, radius_bins=True, particle_ids=ids, errorbars=False)
plt.subplots_adjust(left=0.05)
plt.savefig("/Users/lls/Documents/Papers/deep_halos/hist_pred_radii.pdf")

path_av = "/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4.3/"
p_av = np.load(path_av + "predicted_sim_6_epoch_32.npy")
t_av = np.load(path_av + "true_sim_6_epoch_32.npy")

f1, a, m = pp.plot_histogram_predictions(p1, t1, radius_bins=True, particle_ids=ids, errorbars=False, mid=False,
                                         out=False, legend=False, col="C1")
f11, a1, m1 = pp.plot_histogram_predictions(p_av, t_av, radius_bins=True, particle_ids=ids, errorbars=False, fig=f1,
                                            axes=a, mid=False, out=False, col="C0", legend=False)
plt.savefig(path_av + "raw_vs_av_inner.png")

f1, a, m = pp.plot_histogram_predictions(p1, t1, radius_bins=True, particle_ids=ids, errorbars=False, inner=False,
                                         out=False, legend=False, col="C1")
f11, a1, m1 = pp.plot_histogram_predictions(p_av, t_av, radius_bins=True, particle_ids=ids, errorbars=False, fig=f1,
                                            axes=a, inner=False, out=False, col="C0", legend=False)
plt.savefig(path_av + "raw_vs_av_mid.png")

f1, a, m = pp.plot_histogram_predictions(p1, t1, radius_bins=True, particle_ids=ids, errorbars=False, inner=False,
                                         mid=False, legend=False, col="C1")
f11, a1, m1 = pp.plot_histogram_predictions(p_av, t_av, radius_bins=True, particle_ids=ids, errorbars=False, fig=f1,
                                            axes=a, inner=False, mid=False, col="C0", legend=False)
plt.savefig(path_av + "raw_vs_av_outer.png")
