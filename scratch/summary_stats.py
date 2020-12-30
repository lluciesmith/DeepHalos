import sys
sys.path.append('/Users/lls/Documents/Projects/DeepHalos')
from utilss import radius_functions_deep as rf
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from mlhalos import parameters
from mlhalos import distinct_colours
import numpy as np
import matplotlib.pyplot as plt


def get_summary_stats_mass_bins_radius_bins(predictions, truth, initial_params=None):
    if initial_params is None:
        ic = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/IC.gadget3",
                                                    final_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/snapshot_099")
    else:
        ic = initial_params

    p1 = predictions
    t1 = truth

    ids = np.loadtxt("/Users/lls/Documents/mlhalos_files/reseed50/CNN_results/reseed_1_random_training_set.txt")
    ids = ids.astype("int")

    r = np.load("/Users/lls/Documents/mlhalos_files/reseed50/radii_properties_ids_random_training_set_above_1e11.npy")

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

    mb = [11, 12, 13, truth_inner.max()]
    stats = rf.return_summary_statistics_three_panels(pred_inner, truth_inner, den_mid, truth_mid, den_outer,
                                                      truth_outer, mb)
    return stats



if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/IC.gadget3",
                                                final_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/snapshot_099")

    p_75 = np.load("/Users/lls/Documents/deep_halos_files/z99/ics_res75/predicted1_65.npy")
    t_75 = np.load("/Users/lls/Documents/deep_halos_files/z99/ics_res75/true1_65.npy")

    p_51 = np.load("/Users/lls/Documents/mlhalos_files/reseed50/CNN_results/predicted1_60.npy")
    t_51 = np.load("/Users/lls/Documents/mlhalos_files/reseed50/CNN_results/true1_60.npy")

    p_121 = np.load("/Users/lls/Documents/deep_halos_files/z99/ics_res121/predicted1_30.npy")
    t_121 = np.load("/Users/lls/Documents/deep_halos_files/z99/ics_res121/true1_30.npy")


    stats_75 = get_summary_stats_mass_bins_radius_bins(p_75, t_75, initial_params=ic)
    stats_51 = get_summary_stats_mass_bins_radius_bins(p_51, t_51, initial_params=ic)
    stats_121 = get_summary_stats_mass_bins_radius_bins(p_121, t_121, initial_params=ic)

    color = distinct_colours.get_distinct(4)
    c0 = color[0]
    c1 = color[1]
    c2 = color[3]

    rf.plot_stuff([stats_51, stats_75, stats_121], c0, c1, c2, figsize=(10, 5.1), col_truth="dimgrey")
