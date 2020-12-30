import sys
sys.path.append('/Users/lls/Documents/Projects/DeepHalos')
from utilss import radius_functions_deep as rf
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from mlhalos import distinct_colours
import numpy as np
import matplotlib.pyplot as plt


def get_GBT_predictions():
    test_set_ids = np.load("/Users/lls/Documents/mlhalos_files/testing_ids.npy")
    radii_properties_testing_ids = np.load("/Users/lls/Documents/mlhalos_files/"
                                           "correct_radii_prop_testing_ids_valid_mass_range.npy")

    path = "/Users/lls/Documents/mlhalos_files/LightGBM/CV_only_reseed_sim/"
    truth = np.load(path + "/truth_shear_den_test_set.npy")
    shear = np.load(path + "/predicted_shear_den_test_set.npy")

    bins_valid = np.array([11.42095852, 11.62095852, 11.82095852, 12.02095852, 12.22095852, 12.42095852, 12.62095852,
                           12.82095852, 13.02095852, 13.22095852, 13.42095852])

    assert np.allclose(radii_properties_testing_ids[:, 0].astype("int"),
                       test_set_ids[(truth >= bins_valid.min()) & (truth <= bins_valid.max())])

    ind = (truth >= bins_valid.min()) & (truth <= bins_valid.max())
    truth_valid = truth[ind]
    shear_valid = shear[ind]
    return shear_valid, truth_valid


def plot_histogram_predictions(predictions, truths, particle_ids=None, sim="6", label="new", radius_bins=True,
                               mass_bins=None, r_bins=None,
                              fig=None, axes=None, color="C0"):
    if radius_bins is True:
        if sim == "6":
            r = np.load("/Users/lls/Documents/mlhalos_files/reseed6/reseed6radius_in_halo_particles.npy")[particle_ids]
            r_vir = np.load("/Users/lls/Documents/mlhalos_files/reseed6/reseed6_virial_radius_particles.npy")[particle_ids]
        else:
            r = 0
            r_vir = 0
            ValueError("Sim can only be 6")

        # Ignore particles that don't have a virial radius (because they live in halos that are too small)
        ind = np.where(r_vir != 0)[0]
        r_frac = r[ind] / r_vir[ind]
        p = predictions[ind]
        t = truths[ind]
        if mass_bins is None:
            mass_bins = np.linspace(11, t.max(), 4, endpoint=True)
        if r_bins is None:
            r_bins = np.linspace(-3, 3, 50)

        f, (ax1, ax2, ax3) = rf.plot_radial_cat_in_mass_bins(p, t, r_frac, bins_m=mass_bins, bins_r=r_bins, fig=fig,
                                                                 axes=axes)
    else:
        f, (ax1, ax2, ax3) = rf.plot_diff_predicted_true_mass_ranges(predictions, truths, mass_bins, r_bins,
                                                                     figsize=(11, 5.1),
                                                                     fig=fig, axes=axes, col_truth="dimgrey", lw=1.8,
                                                                     density=True, label=label, color=color)
    return f, (ax1, ax2, ax3)


if __name__ == "__main__":
    # path = "/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_9sims/"
    path = "/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200K/alpha-2.5/"
    num_epoch = '12'

    ids = np.load(path + "../ids_larger_validation_set.npy")
    p = np.load(path + "predicted_sim_6_epoch_" + num_epoch + ".npy")
    t = np.load(path + "true_sim_6_epoch_" + num_epoch + ".npy")

    mass_bins = np.linspace(11, t.max(), 4, endpoint=True)
    r_bins = np.linspace(-3, 3, 50)

    pred_GBT, truth_GBT = get_GBT_predictions()
    mass_bins_GBT = np.linspace(truth_GBT.min(), truth_GBT.max(), 4, endpoint=True)

    f1, (ax11, ax21, ax31) = plot_histogram_predictions(p, t, particle_ids=None, sim="6", label="DL",
                                                        radius_bins=False, mass_bins=mass_bins_GBT, r_bins=r_bins,
                                                        fig=None, axes=None, color="C0")
    f1, (ax11, ax21, ax31) = plot_histogram_predictions(pred_GBT, truth_GBT, particle_ids=None, sim="6", label="GBT",
                                                        radius_bins=False, mass_bins=mass_bins_GBT, r_bins=r_bins,
                                                        fig=f1, axes=(ax11, ax21, ax31), color="dimgrey")

    plt.savefig(path + "comparison_w_GBT_epoch" + num_epoch + ".png")

    f2, (ax12, ax22, ax32) = plot_histogram_predictions(p, t, particle_ids=ids, sim="6", label="DL",
                                                        radius_bins=True,  mass_bins=mass_bins, r_bins=r_bins,
                                                        fig=None, axes=None, color="C0")

    plt.savefig(path + "predictions_radial_analysis_epoch" + num_epoch + ".png")


    # #### PLOT MASS BINS #####
    #
    # color = distinct_colours.get_distinct(4)
    #
    # mass_bins = np.linspace(11, t.max(), 4, endpoint=True)
    # r_bins = np.linspace(-3, 3, 50)
    #
    # f, (ax1, ax2, ax3) = rf.plot_diff_predicted_true_mass_ranges(p, t, mass_bins, r_bins, figsize=(11, 5.1),
    #                                                              col_truth="dimgrey", lw=1.8, density=True, label=label1)
    # f, (ax1, ax2, ax3) = rf.plot_diff_predicted_true_mass_ranges(p_l2, t_l2, mass_bins, r_bins, figsize=(11, 5.1),
    #                                                              col_truth="dimgrey", lw=1.8, density=True,
    #                                                              fig=f, axes=(ax1, ax2, ax3), color="C1", label=label2)
    # plt.yscale("log")
    # plt.subplots_adjust(top=0.93, left=0.1)
