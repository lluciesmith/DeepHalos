import numpy as np
from plots import plots_for_predictions as pp
from utilss import distinct_colours as dc
import matplotlib.pyplot as plt

c = dc.get_distinct(4)

path = '/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'
p1 = np.load(path + "seed_20/predicted_sim_6_epoch_09.npy")
t1 = np.load(path + "seed_20/true_sim_6_epoch_09.npy")

p_big = np.load("/Users/luisals/Projects/DLhalos/bigbox/raw/predicted_sim_L200_N1024_genetIC3_epoch_10.npy")
t_big = np.load("/Users/luisals/Projects/DLhalos/bigbox/raw/true_sim_L200_N1024_genetIC3_epoch_10.npy")


path_av = "/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4.3/"
p_av = np.load(path_av + "predicted_sim_6_epoch_32.npy")
t_av = np.load(path_av + "true_sim_6_epoch_32.npy")

p_av_big = np.load("/Users/luisals/Projects/DLhalos/bigbox/avg/predicted_sim_L200_N1024_genetIC3_epoch_18.npy")
t_av_big = np.load("/Users/luisals/Projects/DLhalos/bigbox/avg/true_sim_L200_N1024_genetIC3_epoch_18.npy")

# Raw-density case
f1, a, m = pp.plot_histogram_predictions(p1, t1, radius_bins=False, particle_ids=None, errorbars=False,
                                         label=r"$L_\mathrm{box}=50 \, \mathrm{Mpc} \,/ \,h$", color="C0")
f11, a1, m1 = pp.plot_histogram_predictions(p_big, t_big, radius_bins=False, particle_ids=None, errorbars=False, fig=f1,
                                            axes=a, color="C1", label=r"$L_\mathrm{box}=200 \, \mathrm{Mpc} \,/ \,h$")
a1[0].set_ylabel(r"$n_{\mathrm{particles}}$", fontsize=16)
[a.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$", fontsize=16) for a in a1]
plt.savefig("/Users/lls/Documents/Papers/dlhalos_paper/small_vs_large_box.pdf")


# Averaged-density case

f1, a, m = pp.plot_histogram_predictions(p_av, t_av, radius_bins=False, particle_ids=None, errorbars=False,
                                         label=r"$L_\mathrm{box}=50 \, \mathrm{Mpc} \,/ \,h$", color="C0")
f11, a1, m1 = pp.plot_histogram_predictions(p_av_big, t_av_big, radius_bins=False, particle_ids=None, errorbars=False, fig=f1,
                                            axes=a, color="C1", label=r"$L_\mathrm{box}=200 \, \mathrm{Mpc} \,/ \,h$")
a1[0].set_ylabel(r"$n_{\mathrm{particles}}$", fontsize=16)
[a.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$", fontsize=16) for a in a1]
plt.savefig("/Users/luisals/Documents/Papers/dlhalos_paper/averaged_small_vs_large_box.pdf")

# Averaged-density case

f1, a, m = pp.plot_histogram_predictions(p_big, t_big, radius_bins=False, particle_ids=None, errorbars=False,
                                         label="Raw density", color="C0")
f11, a1, m1 = pp.plot_histogram_predictions(p_av_big, t_av_big, radius_bins=False, particle_ids=None, errorbars=False, fig=f1,
                                            axes=a, color="C1", label="Averaged density")
a1[0].set_ylabel(r"$n_{\mathrm{particles}}$", fontsize=16)
[a.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$", fontsize=16) for a in a1]
plt.savefig("/Users/luisals/Documents/Papers/dlhalos_paper/raw_vs_averaged_large_box.pdf")
