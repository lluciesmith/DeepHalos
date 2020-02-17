import numpy as np
import matplotlib.pyplot as plt

def sort_ids_truth_pred(particle_ids, truth, predicted, ic_params):
    h = []
    for i in particle_ids:
        h.append(np.log10(ic_params.halo[ic_params.final_snapshot['grp'][i]]['mass'].sum()))
    har = np.array(h)
    ind_sort_ids = np.argsort(har)
    ids_sort = particle_ids[ind_sort_ids]

    ind_sort = np.argsort(truth)
    truth_sort = truth[ind_sort]
    pred_sort = predicted[ind_sort]

    assert np.allclose(truth_sort, har[ind_sort_ids])

    return np.sort(ids_sort), truth_sort[np.argsort(ids_sort)], pred_sort[np.argsort(ids_sort)]

def get_indices_particles_in_bin_limits(radii_properties, bin_lower_lim=0, bin_upper_lim=100, return_ids=False):
    ids = radii_properties[:, 0]
    fraction = radii_properties[:, 2]
    ind_bin = np.where((fraction > bin_lower_lim) & (fraction <= bin_upper_lim))[0]
    if return_ids is True:
        return ids[ind_bin]
    else:
        return ind_bin


def plot_radial_ranges_in_3_mass_bins(den_inner, truth_inner, den_mid, truth_mid, den_outer, truth_outer,
                                      bins_radial_distributions, mass_bins,
                                      col_in, col_mid, col_out, figsize=(10, 5.2), col_truth="dimgrey", lw=1.8,
                                      density=True,  fontsize=15):
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True, sharex=True)
    ax1.axvline(x=mb[0], color=col_truth)
    ax1.axvline(x=mb[1], color=col_truth)
    # _ = ax1.hist(truth_inner[(truth_inner > mb[0]) & (truth_inner < mb[1])], bins=b, histtype="step",
    #              density=density, lw=lw, color=col_truth)
    _ = ax1.hist(den_inner[(truth_inner > mass_bins[0]) & (truth_inner < mass_bins[1])], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_in)
    _ = ax1.hist(den_mid[(truth_mid > mass_bins[0]) & (truth_mid < mass_bins[1])], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_mid)
    _ = ax1.hist(den_outer[(truth_outer > mass_bins[0]) & (truth_outer < mass_bins[1])], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_out)

    ax2.axvline(x=mb[1], color=col_truth)
    ax2.axvline(x=mb[2], color=col_truth)
    # _ = ax2.hist(truth_inner[(truth_inner > mass_bins[1]) & (truth_inner < mass_bins[2])], bins=b, histtype="step",
    #              density=density, lw=lw, color=col_truth)
    _ = ax2.hist(den_inner[(truth_inner > mass_bins[1]) & (truth_inner < mass_bins[2])], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_in)
    _ = ax2.hist(den_mid[(truth_mid > mass_bins[1]) & (truth_mid < mass_bins[2])], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_mid)
    _ = ax2.hist(den_outer[(truth_outer > mass_bins[1]) & (truth_outer < mass_bins[2])], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_out)

    ax3.axvline(x=mb[2], color=col_truth)
    ax3.axvline(x=mb[3], color=col_truth)
    # _ = ax3.hist(truth_inner[(truth_inner > mass_bins[2]) & (truth_inner < mass_bins[3])], bins=bins_radial_distributions,
    #              histtype="step", density=density, lw=lw, color=col_truth)
    aa = ax3.hist(den_inner[(truth_inner > mass_bins[2]) & (truth_inner < mass_bins[3])], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_in)
    bb = ax3.hist(den_mid[(truth_mid > mass_bins[2]) & (truth_mid < mass_bins[3])], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_mid)
    dd = ax3.hist(den_outer[(truth_outer > mass_bins[2]) & (truth_outer < mass_bins[3])], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_out)

    plt.subplots_adjust(wspace=0, bottom=0.15, left=0.08)
    plt.figlegend((aa[2][0], bb[2][0], dd[2][0]),
                  (r"$r/\mathrm{r_{vir}} \leq %.1f $" % i1,
                   r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (m0, m1),
                   # r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (o0, o1),
                   r"$r/\mathrm{r_{vir}} > %.1f $" % o0,
                   ),
                   loc='upper center', bbox_to_anchor=(0.09, 0.45, 0.5, 0.5), fontsize=fontsize-1, framealpha=1)

    ax1.set_ylabel(r"$n_{\mathrm{particles}}$", fontsize=fontsize+1)
    ax1.set_xlabel(r"$\log(M_{\mathrm{predicted}}/\mathrm{M_\odot})$", fontsize=fontsize)
    ax2.set_xlabel(r"$\log(M_{\mathrm{predicted}}/\mathrm{M_\odot})$", fontsize=fontsize)
    ax3.set_xlabel(r"$\log(M_{\mathrm{predicted}}/\mathrm{M_\odot})$", fontsize=fontsize)


def plot_diff_predicted_true_radial_ranges(den_inner, truth_inner, den_mid, truth_mid, den_outer, truth_outer,
                                          bins_radial_distributions, mass_bins,
                                          col_in, col_mid, col_out, i1, m0, m1, o0, figsize=(10, 5.1),
                                           col_truth="dimgrey", lw=1.8,
                                          density=True, fontsize=15):
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True, sharex=True)

    ax1.axvline(x=0, color=col_truth, ls="--")
    ax2.axvline(x=0, color=col_truth, ls="--")
    ax3.axvline(x=0, color=col_truth, ls="--")

    low_mass_inner = (truth_inner >= mass_bins[0]) & (truth_inner < mass_bins[1])
    low_mass_mid = (truth_mid >= mass_bins[0]) & (truth_mid < mass_bins[1])
    low_mass_outer = (truth_outer >= mass_bins[0]) & (truth_outer < mass_bins[1])

    _ = ax1.hist(den_inner[low_mass_inner] - truth_inner[low_mass_inner], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_in)
    _ = ax1.hist(den_mid[low_mass_mid] - truth_mid[low_mass_mid], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_mid)
    _ = ax1.hist(den_outer[low_mass_outer] - truth_outer[low_mass_outer], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_out)
    ax1.set_title(r"$ %.2f \leq \log(M_{\mathrm{true}}) \leq %.2f$" % (mass_bins[0], mass_bins[1]))

    mid_mass_inner = (truth_inner >= mass_bins[1]) & (truth_inner < mass_bins[2])
    mid_mass_mid = (truth_mid >= mass_bins[1]) & (truth_mid < mass_bins[2])
    mid_mass_outer = (truth_outer >= mass_bins[1]) & (truth_outer < mass_bins[2])
    _ = ax2.hist(den_inner[mid_mass_inner] - truth_inner[mid_mass_inner], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_in)
    _ = ax2.hist(den_mid[mid_mass_mid] - truth_mid[mid_mass_mid], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_mid)
    _ = ax2.hist(den_outer[mid_mass_outer] - truth_outer[mid_mass_outer], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_out)
    ax2.set_title(r"$%.2f \leq \log(M_{\mathrm{true}}) \leq %.2f$" % (mass_bins[1], mass_bins[2]))

    high_mass_inner = (truth_inner >= mass_bins[2]) & (truth_inner <= mass_bins[3])
    high_mass_mid = (truth_mid >= mass_bins[2]) & (truth_mid <= mass_bins[3])
    high_mass_outer = (truth_outer >= mass_bins[2]) & (truth_outer <= mass_bins[3])
    aa = ax3.hist(den_inner[high_mass_inner] - truth_inner[high_mass_inner], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_in)
    bb = ax3.hist(den_mid[high_mass_mid] - truth_mid[high_mass_mid], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_mid)
    dd = ax3.hist(den_outer[high_mass_outer] - truth_outer[high_mass_outer], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_out)
    ax3.set_title(r"$\log(M_{\mathrm{true}}) \geq %.2f$" % mass_bins[2])

    plt.subplots_adjust(wspace=0, bottom=0.14, left=0.08)
    plt.figlegend((aa[2][0], bb[2][0], dd[2][0]),
                  (r"$r/\mathrm{r_{vir}} \leq %.1f $" % i1,
                   r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (m0, m1),
                   # r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (o0, o1),
                   r"$r/\mathrm{r_{vir}} > %.1f $" % o0,
                   ),
                   loc='upper center', bbox_to_anchor=(0.12, 0.45, 0.5, 0.5), framealpha=1)

    ax1.set_ylabel(r"$n_{\mathrm{particles}}$")
    ax1.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax2.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax3.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax3.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])

    # ax1.text(0.2, 0.8, "Low-mass \n haloes", horizontalalignment='center', verticalalignment='center',
    #          fontweight='bold', transform=ax1.transAxes)
    # ax2.text(0.8, 0.8, "Mid-mass \n haloes", horizontalalignment='center', verticalalignment='center',
    #          fontweight='bold', transform=ax2.transAxes)
    # ax3.text(0.8, 0.8, "High-mass \n haloes", horizontalalignment='center', verticalalignment='center',
    #          fontweight='bold', transform=ax3.transAxes)


def plot_diff_predicted_true_subhalos(den_inner, truth_inner, den_mid, truth_mid, den_outer, truth_outer,
                                          bins_radial_distributions, mass_bins,
                                          col_in, col_mid, col_out, figsize=(10, 5.1), col_truth="dimgrey", lw=1.8,
                                          density=True, fontsize=15):
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True, sharex=True)

    ax1.axvline(x=0, color=col_truth, ls="--")
    ax2.axvline(x=0, color=col_truth, ls="--")
    ax3.axvline(x=0, color=col_truth, ls="--")

    _ = ax1.hist(den_inner[(truth_inner > mass_bins[0]) & (truth_inner < mass_bins[1])]
                 - truth_inner[(truth_inner > mass_bins[0]) & (truth_inner < mass_bins[1])], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_in)
    _ = ax1.hist(den_mid[(truth_mid > mass_bins[0]) & (truth_mid < mass_bins[1])]
                 - truth_mid[(truth_mid > mass_bins[0]) & (truth_mid < mass_bins[1])], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_mid)
    _ = ax1.hist(den_outer[(truth_outer > mass_bins[0]) & (truth_outer < mass_bins[1])]
                 - truth_outer[(truth_outer > mass_bins[0]) & (truth_outer < mass_bins[1])],
                 bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_out)

    mid_mass_inner = (truth_inner > mass_bins[1]) & (truth_inner < mass_bins[2])
    mid_mass_mid = (truth_mid > mass_bins[1]) & (truth_mid < mass_bins[2])
    mid_mass_outer = (truth_outer > mass_bins[1]) & (truth_outer < mass_bins[2])
    _ = ax2.hist(den_inner[mid_mass_inner] - truth_inner[mid_mass_inner], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_in)
    _ = ax2.hist(den_mid[mid_mass_mid] - truth_mid[mid_mass_mid], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_mid)
    _ = ax2.hist(den_outer[mid_mass_outer] - truth_outer[mid_mass_outer], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_out)

    high_mass_inner = (truth_inner > mass_bins[2]) & (truth_inner < mass_bins[3])
    high_mass_mid = (truth_mid > mass_bins[2]) & (truth_mid < mass_bins[3])
    high_mass_outer = (truth_outer > mass_bins[2]) & (truth_outer < mass_bins[3])
    aa = ax3.hist(den_inner[high_mass_inner] - truth_inner[high_mass_inner], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_in)
    bb = ax3.hist(den_mid[high_mass_mid] - truth_mid[high_mass_mid], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_mid)
    dd = ax3.hist(den_outer[high_mass_outer] - truth_outer[high_mass_outer], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_out)

    plt.subplots_adjust(wspace=0, bottom=0.14, left=0.08)
    plt.figlegend((aa[2][0], bb[2][0], dd[2][0]),
                  (r"$r/\mathrm{r_{vir}} \leq %.1f $" % i1,
                   r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (m0, m1),
                   # r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (o0, o1),
                   r"$r/\mathrm{r_{vir}} > %.1f $" % o0,
                   ),
                   loc='upper center', bbox_to_anchor=(0.12, 0.45, 0.5, 0.5), framealpha=1)

    ax1.set_ylabel(r"$n_{\mathrm{particles}}$")
    ax1.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax2.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax3.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")

    ax1.text(0.2, 0.8, "Low-mass \n haloes", horizontalalignment='center', verticalalignment='center',
             fontweight='bold', transform=ax1.transAxes)
    ax2.text(0.8, 0.8, "Mid-mass \n haloes", horizontalalignment='center', verticalalignment='center',
             fontweight='bold', transform=ax2.transAxes)
    ax3.text(0.8, 0.8, "High-mass \n haloes", horizontalalignment='center', verticalalignment='center',
             fontweight='bold', transform=ax3.transAxes)


def plot_radial_range_only_high_mass(den_inner, truth_inner, den_mid, truth_mid, den_outer, truth_outer,
                                          bins_radial_distributions, mass_bins,
                                          col_in, col_mid, col_out, i1, m0, m1, o0, figsize=(10, 5.1),
                                           col_truth="dimgrey", lw=1.8,
                                          density=True, fontsize=15):
    f, ax3 = plt.subplots(nrows=1, ncols=1)
    ax3.axvline(x=0, color=col_truth, ls="--")

    high_mass_inner = (truth_inner >= mass_bins[2]) & (truth_inner <= mass_bins[3])
    high_mass_mid = (truth_mid >= mass_bins[2]) & (truth_mid <= mass_bins[3])
    high_mass_outer = (truth_outer >= mass_bins[2]) & (truth_outer <= mass_bins[3])
    aa = ax3.hist(den_inner[high_mass_inner] - truth_inner[high_mass_inner], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_in)
    bb = ax3.hist(den_mid[high_mass_mid] - truth_mid[high_mass_mid], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_mid)
    dd = ax3.hist(den_outer[high_mass_outer] - truth_outer[high_mass_outer], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_out)
    ax3.set_title(r"$\log(M_{\mathrm{true}}) \geq %.2f$" % mass_bins[2])

    plt.subplots_adjust(wspace=0, bottom=0.14, left=0.08)
    plt.figlegend((aa[2][0], bb[2][0], dd[2][0]),
                  (r"$r/\mathrm{r_{vir}} \leq %.1f $" % i1,
                   r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (m0, m1),
                   # r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (o0, o1),
                   r"$r/\mathrm{r_{vir}} > %.1f $" % o0,
                   ),
                   loc='best',
                  bbox_to_anchor=(0.12, 0.45, 0.4, 0.4),
                  framealpha=1, fontsize=16)

    ax3.set_ylabel(r"$n_{\mathrm{particles}}$")
    ax3.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax3.set_xticks([-3, -2, -1, 0, 1, 2, 3])