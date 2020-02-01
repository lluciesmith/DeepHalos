import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from mlhalos import parameters
from mlhalos import distinct_colours
import numpy as np
import matplotlib.pyplot as plt
from utils import radius_functions_deep as rf


if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/IC.gadget3",
                                                final_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/snapshot_099")

    ids = np.loadtxt("/Users/lls/Documents/mlhalos_files/reseed50/CNN_results/reseed_1_random_training_set.txt")
    ids = ids.astype("int")

    r = np.load("/Users/lls/Documents/mlhalos_files/reseed50/radii_properties_ids_random_training_set_above_1e11.npy")

    t1 = np.load("/Users/lls/Documents/deep_halos_files/z0/correct_ordering/truth1_80.npy")
    p1 = np.load("/Users/lls/Documents/deep_halos_files/z0/correct_ordering/pred1_80.npy")

    halo_ids = ic.final_snapshot[ids]['grp']
    ids_above_1e11 = ids[halo_ids <= 5300]
    p1_above_1e11 = p1[halo_ids <= 5300]
    t1_above_1e11 = t1[halo_ids <= 5300]

    i1 = 0.2
    m0 = 0.4
    m1 = 0.6
    o0 = 0.8
    o1 = 100
    ind_inner = rf.get_indices_particles_in_bin_limits(r, bin_lower_lim=0.0000000001, bin_upper_lim=i1)
    ind_mid = rf.get_indices_particles_in_bin_limits(r, bin_lower_lim=m0, bin_upper_lim=m1)
    ind_outer = rf.get_indices_particles_in_bin_limits(r, bin_lower_lim=o0, bin_upper_lim=o1)

    ###### INNER #####
    truth_inner = t1_above_1e11[ind_inner]
    pred_inner = p1_above_1e11[ind_inner]

    ###### MID #####

    truth_mid = t1_above_1e11[ind_mid]
    den_mid = p1_above_1e11[ind_mid]

    ###### OUTER #####
    truth_outer = t1_above_1e11[ind_outer]
    den_outer = p1_above_1e11[ind_outer]

    #### PLOT #####

    color = distinct_colours.get_distinct(4)
    c0 = color[0]
    c1 = color[1]
    c2 = color[3]
    lw = 1.8
    density = False

    mb = [truth_inner.min(), 12, 13, truth_inner.max()]
    b = np.linspace(-4, 4, 50)
    rf.plot_diff_predicted_true_radial_ranges(pred_inner, truth_inner, den_mid, truth_mid, den_outer, truth_outer,
                                              b, mb, c0, c1, c2, i1, m0, m1, o0,
                                              lw=lw, density=density, figsize=(12, 5.2), fontsize=15)
    plt.yscale("log")
    plt.subplots_adjust(top=0.93)
    plt.savefig("/Users/lls/Documents/deep_halos_files/z0/predictions_mass_and_radius_bins.png")





