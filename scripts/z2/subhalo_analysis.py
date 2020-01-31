import sys
import numpy as np
sys.path.append('/Users/lls/Documents/Projects/DeepHalos/'); from utils import radius_functions_deep as rf
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from mlhalos import parameters
from mlhalos import distinct_colours
import matplotlib.pyplot as plt


def make_grp_subhalo_catalogue(snapshot, subhalo_catalogue):
    ids = snapshot['iord']
    assert np.allclose(ids, np.sort(ids)), "This code breaks if the particle ids are not sorted"

    h = snapshot.halos(make_grp=True)
    subhalo_ids = np.concatenate([h[i].properties['children'][1:] for i in range(len(h))])

    subh_grp = np.ones(len(ids),) * -1
    for subh_id in subhalo_ids:
        ids_i = subhalo_catalogue[subh_id]['iord']
        subh_grp[ids_i] = subh_id

    snapshot['subh_grp'] = subh_grp.astype("int")
    return snapshot

if __name__ == "__main__":

    ic = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/IC.gadget3",
                                                final_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/snapshot_099")
    subh = ic.final_snapshot.halos(subs=True)
    make_grp_subhalo_catalogue(ic.final_snapshot, subh)

    ids = np.loadtxt("/Users/lls/Documents/mlhalos_files/reseed50/CNN_results/reseed_1_random_training_set.txt")
    ids = ids.astype("int")

    subh_mass_ids = np.ones(len(ids),)* -1
    for num, i in enumerate(ids):
        subhalo_id = ic.final_snapshot['subh_grp'][i]
        if subhalo_id != -1:
            subh_mass_ids[num] = np.log10(subh[ic.final_snapshot['subh_grp'][i]]['mass'].sum())

    t1 = np.load("/Users/lls/Documents/deep_halos_files/z0/correct_ordering/truth1_80.npy")
    p1 = np.load("/Users/lls/Documents/deep_halos_files/z0/correct_ordering/pred1_80.npy")
    r = np.load("/Users/lls/Documents/mlhalos_files/reseed50/radii_properties_ids_random_training_set_above_1e11.npy")

    halo_ids = ic.final_snapshot[ids]['grp']
    ids_above_1e11 = ids[halo_ids <= 5300]
    p1_above_1e11 = p1[halo_ids <= 5300]
    t1_above_1e11 = t1[halo_ids <= 5300]
    subh_mass_ids_above_1e11 = subh_mass_ids[halo_ids <= 5300]

    ind_inner = rf.get_indices_particles_in_bin_limits(r, bin_lower_lim=0.0000000001, bin_upper_lim=0.2)
    ind_mid = rf.get_indices_particles_in_bin_limits(r, bin_lower_lim=0.4, bin_upper_lim=0.6)
    ind_outer = rf.get_indices_particles_in_bin_limits(r, bin_lower_lim=0.8, bin_upper_lim=100)

    ###### INNER #####
    truth_inner = t1_above_1e11[ind_inner]
    pred_inner = p1_above_1e11[ind_inner]
    subh_mass_inner = subh_mass_ids_above_1e11[ind_inner]

    ###### MID #####

    truth_mid = t1_above_1e11[ind_mid]
    pred_mid = p1_above_1e11[ind_mid]
    subh_mass_mid = subh_mass_ids_above_1e11[ind_mid]

    ###### OUTER #####
    truth_outer = t1_above_1e11[ind_outer]
    pred_outer = p1_above_1e11[ind_outer]
    subh_mass_outer = subh_mass_ids_above_1e11[ind_outer]

    # Look at particles in the outskirts of high mass halos. Then take those that have the worst predictions,
    # i.e. log(M_predicted) - log(M_truth) >= 1. How do the predictions correlate with the mass of the subhalo in
    # which they belong?

    ind_high_mass = (truth_outer >= 13)
    truth_outer_high_mass = truth_outer[ind_high_mass]
    pred_outer_high_mass = pred_outer[ind_high_mass]
    subh_mass_outer_high_mass = subh_mass_outer[ind_high_mass]

    ind_wrong = np.where(abs(truth_outer_high_mass - pred_outer_high_mass) >= 1)[0]

    plt.scatter(subh_mass_outer_high_mass[ind_wrong], pred_outer_high_mass[ind_wrong], s=4, alpha=0.8,
                label=r"$|\log(M_\mathrm{true}/M_\mathrm{predicted})|\geq 1$")
    plt.legend(loc=2)
    plt.plot([10, 15], [10, 15], color="grey")
    plt.title(r"Outskirt particles($r/r_\mathrm{vir}>0.8$) in halos $\log(M_\mathrm{halo})\geq 13$", fontsize=16)
    plt.xlabel(r"$\log(M_\mathrm{subhalo}/M_\odot)$")
    plt.ylabel(r"$\log(M_\mathrm{predicted}/M_\odot)$")
    plt.subplots_adjust(bottom=0.13, top=0.92)
    plt.savefig("/Users/lls/Documents/deep_halos_files/z0/outskirt_predictions_vs_subhalo_mass.png")

    # Look at particles in the outskirts of high mass halos.
    # What is their distribution of log(M_truth/M_predicted) for all particles vs those not in subhalos?

    ind_insub = np.where(subh_mass_outer_high_mass != -1)[0]
    ind_not_insubh = np.where(subh_mass_outer_high_mass == -1)[0]
    b = np.linspace(-4, 4, 30)
    color = distinct_colours.get_distinct(4)
    plt.hist(pred_outer_high_mass - truth_outer_high_mass, bins=b, histtype="step", color=color[3], lw=2,
             label="all outskirt particles")
    plt.hist((pred_outer_high_mass - truth_outer_high_mass)[ind_insub], bins=b, histtype="step", lw=2,
             label="in subhalos")
    plt.hist((pred_outer_high_mass - truth_outer_high_mass)[ind_not_insubh], bins=b, histtype="step", lw=2,
             label="not in subhalos")


