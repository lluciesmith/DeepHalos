import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath("./DeepHalos/"))))
from utilss import radius_functions_deep as rf
from plots import predictions_functions as pf
from dlhalos_code import loss_functions as lf
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


def plot_histogram_predictions(predictions, truths, particle_ids=None, sim="6", label="new", radius_bins=True,
                               mass_bins=None, r_bins=None, fig=None, axes=None, color="C0", density=True,
                               errorbars=True, inner=True, mid=True, out=True, legend=True, col=None, alpha=None):
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
                                                             inner=inner, mid=mid, out=out, legend=legend, col=col,
                                                             axes=axes, density=density, errorbars=errorbars)
    else:
        if mass_bins is None:
            mass_bins = np.linspace(11, 13.4, 4, endpoint=True)
        if r_bins is None:
            r_bins = np.linspace(-3, 3, 50)

        f, (ax1, ax2, ax3) = rf.plot_diff_predicted_true_mass_ranges(predictions, truths, mass_bins, r_bins,
                                                                     figsize=(11, 5.1),
                                                                     fig=fig, axes=axes, col_truth="dimgrey", lw=1.8,
                                                                     density=True, label=label, color=color)
    return f, (ax1, ax2, ax3), mass_bins


def get_truth_predictions_GBT():
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
    return truth_valid, shear_valid


def predictions_GBT_mass_bins(fig=None, axes=None, color="C1"):
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

    mass_bins = np.linspace(truth_valid.min(), truth_valid.max(), 4, endpoint=True)
    r_bins = np.linspace(-3, 3, 50)
    f, axes = rf.plot_diff_predicted_true_mass_ranges(shear_valid, truth_valid, mass_bins, r_bins, figsize=(11, 5.1),
                                                                 col_truth="dimgrey", lw=1.8, density=True, label="GBT",
                                                                 fig=fig, axes=axes, color=color)
    return f, axes, mass_bins


def plot_violin(true, predicted, bins_violin=None,labels=None, return_stats="median",
                fig=None, axes=None, col1="#8C4843", col1_violin="#A0524D"):

    if len(true) == 2:
        if bins_violin is None:
            true_all = np.concatenate(true)
            bins_violin = np.linspace(true_all.min(), true_all.max(), 13)

        col2 = "#406D60"
        col2_violin = "#55917F"

        fig, ax = pf.compare_two_violin_plots(predicted[0], true[0], predicted[1], true[1],
                                              bins_violin, label1=labels[0], label2=labels[1], return_stats=return_stats,
                                              col1=col1, col2=col2, col1_violin=col1_violin, col2_violin=col2_violin,
                                              alpha1=0.3, alpha2=0.3, figsize=(6.9, 5.2),
                                              edge1=col1, edge2=col2)
        plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.48, 0.5, 0.5), framealpha=1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        plt.subplots_adjust(left=0.14)
    else:
        if bins_violin is None:
            bins_violin = np.linspace(true.min(), true.max(), 13)
        fig, ax = pf.violin_plot(predicted, true, bins_violin, path=None, label1=labels, return_stats=return_stats,
                                 figsize=(8, 6), title=None, col1=col1, col1_violin=col1_violin, alpha1=0.3,
                                 fig=fig, axes=axes)
    return fig, ax


def plot_corner(true, predicted):
    import corner
    levels = (0.68, 0.95, 0.997)
    plf = corner.hist2d(true[true >= 11], predicted[true >= 11],
                        levels=levels, bins=48, smooth=True, color="midnightblue")
    #plt.ylim(11.1, 14.5)
    plt.xlabel(r"$\log(M_\mathrm{true}/\mathrm{M_\odot})$")
    plt.ylabel(r"$\log(M_\mathrm{predicted}/\mathrm{M_\odot})$")

    plt.subplots_adjust(bottom=0.14, left=0.15, top=0.92)


def plot_scatter_predictions_radius(p, t, sim="6"):
    if sim == "6":
        ids = np.load(
            "/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/ids_larger_validation_set.npy")
        r = np.load("/Users/lls/Documents/mlhalos_files/reseed6/reseed6radius_in_halo_particles.npy")[ids]
        r_vir = np.load("/Users/lls/Documents/mlhalos_files/reseed6/reseed6_virial_radius_particles.npy")[ids]
        frac_r = r / r_vir

    f, ax = plt.subplots(1, 1)

    plt.axhline(y=0, lw=0.5, color="k")
    plt.scatter(frac_r[t >= 11], (p - t)[t >= 11], c=t[t >= 11], s=0.11, cmap="inferno")
    cmap = plt.colorbar()
    cmap.set_label(r"$\log \left( M_\mathrm{true} \right)$")

    plt.xscale("log")
    plt.xlabel(r"$r/r_\mathrm{vir}$")
    plt.ylabel(r"$\log \left( M_\mathrm{predicted}/M_\mathrm{true}\right)$")
    plt.axvline(x=1, color="grey", ls="--")

    # ax.set_xscale("log")
    # ax.set_xlabel(r"$r/r_\mathrm{vir}$")
    # ax.set_ylabel(r"$\log \left( M_\mathrm{predicted}/M_\mathrm{true}\right)$")
    # ax.axvline(x=1, color="grey", ls="--")
    return f, ax


def plot_likelihood_distribution(p, t, gamma, scaler, bins=None, fig=None, axes=None, color="C0", title=None,
                                 legend=True, slices=None):
    if bins is None:
        bins = np.linspace(-1, 1, 60, endpoint=True)
    p_rescaled = scaler.transform(p.reshape(-1, 1)).flatten()
    t_rescaled = scaler.transform(t.reshape(-1, 1)).flatten()

    if slices is None:
        slices = [-0.9, -0.5, 0, 0.5, 0.9, 0.95]
        # slices = [-0.6, -0.2, 0, 0.5, 0.75, 0.95]

    slices = [(bi-0.01, bi+0.01) for bi in slices]

    L = lf.ConditionalCauchySelectionLoss(gamma=gamma)

    if fig is None:
        fig, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 8))
    axes2 = axes.flatten()

    for i, slice in enumerate(slices):
        ax = axes2[i]

        ind = np.where((p_rescaled >= slice[0]) * (p_rescaled <= slice[1]))[0]
        mean_x = np.mean([slice[0], slice[1]])

        _ = ax.hist(t_rescaled[ind], bins=bins, histtype="step", lw=2, label="$%.2f \leq y \leq %.2f $" % slice,
                    density=True, color=color)

        y_pred = np.repeat(mean_x, len(bins))
        liklih = np.exp(- L.loss(bins.reshape(-1, 1), y_pred.reshape(-1, 1)))
        ___ = ax.plot(bins, liklih, color="k")

        ax.axvline(x=np.mean([slice[0], slice[1]]), color="k")
        if legend is True:
            if i > 1:
                ax.legend(loc=2)
            else:
                ax.legend(loc="best")
        ax.get_yaxis().set_ticks([])

    plt.subplots_adjust(left=0.05, bottom=0.12, wspace=0, hspace=0)
    fig.text(0.5, 0.01, r"$d$")
    fig.text(0.01, 0.5, r"P$\left( d | y(\mathbf{w}) \right)$", rotation='vertical', va='center')
    if title is not None:
        fig.suptitle(title)
        plt.subplots_adjust(left=0.05, bottom=0.12, wspace=0, hspace=0, top=0.92)
    return fig, axes


def plot_predicted_in_fine_mass_bins(pred1, truth1, pred2, truth2, mass_bins, bins, label1="averaged-density",
                                     label2="raw-density", color1="C0", color2="C1", figsize=(16, 8)):
    f, axes = plt.subplots(4, 4, figsize=figsize, sharex=True)
    axes = axes.flatten()
    plt.subplots_adjust(bottom=0.1, hspace=0, left=0.05, wspace=0, top=0.92)

    for ax in axes:
        ax.set_yticks([])

    for i in range(len(mass_bins) - 1):
        ax = axes[i]
        ind1 = (truth1>= mass_bins[i]) & (truth1<=mass_bins[i+1])
        ind2 = (truth2 >= mass_bins[i]) & (truth2 <= mass_bins[i + 1])
        if i ==1:
            _ = ax.hist(pred1[ind1], density=True, bins=bins, histtype="step", lw=True, color=color1, label=label1)
            __ = ax.hist(pred2[ind2], density=True, bins=bins, histtype="step", lw=True, color=color2, label=label2)
        _ = ax.hist(pred1[ind1], density=True, bins=bins, histtype="step", lw=True, color=color1)
        __ = ax.hist(pred2[ind2], density=True, bins=bins, histtype="step", lw=True, color=color2)
        _ = ax.axvline(x=mass_bins[i], color="dimgrey")
        _ = ax.axvline(x=mass_bins[i+1], color="dimgrey")

    ax.set_xticks([10, 11, 12, 13, 14])
    axes[1].legend(ncol=2, bbox_to_anchor=(0.2, 1.4), fontsize=14,loc='upper left')
    f.text(0.45, 0.01, r"$\log (M_\mathrm{predicted})$")
    return f, axes
