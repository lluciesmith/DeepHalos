import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

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
    print(stats.describe(den_outer[high_mass_outer] - truth_outer[high_mass_outer]))
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


def return_summary_statistics_three_panels(den_inner, truth_inner, den_mid, truth_mid, den_outer, truth_outer,
                                          mass_bins):

    low_mass_inner = (truth_inner >= mass_bins[0]) & (truth_inner < mass_bins[1])
    low_mass_mid = (truth_mid >= mass_bins[0]) & (truth_mid < mass_bins[1])
    low_mass_outer = (truth_outer >= mass_bins[0]) & (truth_outer < mass_bins[1])

    d_in_low = stats.describe(den_inner[low_mass_inner] - truth_inner[low_mass_inner])
    d_mid_low = stats.describe(den_mid[low_mass_mid] - truth_mid[low_mass_mid])
    d_out_low = stats.describe(den_outer[low_mass_outer] - truth_outer[low_mass_outer])

    mid_mass_inner = (truth_inner >= mass_bins[1]) & (truth_inner < mass_bins[2])
    mid_mass_mid = (truth_mid >= mass_bins[1]) & (truth_mid < mass_bins[2])
    mid_mass_outer = (truth_outer >= mass_bins[1]) & (truth_outer < mass_bins[2])

    d_in_mid = stats.describe(den_inner[mid_mass_inner] - truth_inner[mid_mass_inner])
    d_mid_mid = stats.describe(den_mid[mid_mass_mid] - truth_mid[mid_mass_mid])
    d_out_mid = stats.describe(den_outer[mid_mass_outer] - truth_outer[mid_mass_outer])

    high_mass_inner = (truth_inner >= mass_bins[2]) & (truth_inner <= mass_bins[3])
    high_mass_mid = (truth_mid >= mass_bins[2]) & (truth_mid <= mass_bins[3])
    high_mass_outer = (truth_outer >= mass_bins[2]) & (truth_outer <= mass_bins[3])

    d_in_high = stats.describe(den_inner[high_mass_inner] - truth_inner[high_mass_inner])
    d_mid_high = stats.describe(den_mid[high_mass_mid] - truth_mid[high_mass_mid])
    d_out_high = stats.describe(den_outer[high_mass_outer] - truth_outer[high_mass_outer])

    diction = {}

    diction['low-mass'] = {'inner': {'mean': d_in_low.mean, 'var': d_in_low.variance, 'skew':d_in_low.skewness},
                     'mid': {'mean': d_mid_low.mean, 'var': d_mid_low.variance, 'skew':d_mid_low.skewness},
                     'outer': {'mean': d_out_low.mean, 'var': d_out_low.variance, 'skew':d_out_low.skewness}}
    diction['mid-mass'] = {'inner': {'mean': d_in_mid.mean, 'var': d_in_mid.variance, 'skew': d_in_mid.skewness},
                'mid': {'mean': d_mid_mid.mean, 'var': d_mid_mid.mean, 'skew': d_mid_mid.skewness},
                'outer': {'mean': d_out_mid.mean, 'var': d_out_mid.variance, 'skew': d_out_mid.skewness}}
    diction['high-mass'] = {'inner': {'mean': d_in_high.mean, 'var': d_in_high.variance, 'skew': d_in_high.skewness},
                 'mid': {'mean': d_mid_high.mean, 'var': d_mid_high.variance, 'skew': d_mid_high.skewness},
                 'outer': {'mean': d_out_high.mean, 'var': d_out_high.variance, 'skew': d_out_high.skewness}}

    # d1 = pd.DataFrame.from_dict({(i, j): diction[i][j]
    #                         for i in diction.keys()
    #                         for j in diction[i].keys()},
    #                        orient='columns')
    #
    # d = [dict(selector="th", props=[('text-align', 'center')]),
    #      dict(selector="td", props=[('text-align', 'center')])]
    # d1.style.set_properties(**{'width': '100cm', 'text-align': 'center'}).set_table_styles([d])
    return diction


def plot_stuff(dictions, col_in, col_mid, col_out, labels=[r"$51^3$", r"$75^3$", r"$121^3$"], figsize=(10, 5.1),
               col_truth="dimgrey"):
    f1, axs1 = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True, sharex=True, constrained_layout=True)
    f2, axs2 = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True, sharex=True, constrained_layout=True)
    f3, axs3 = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True, sharex=True, constrained_layout=True)

    figs = (f1, f2, f3)
    axes = (axs1, axs2, axs3)
    cols = (col_in, col_mid, col_out)

    for ax in axes:
        for axi in ax:
            axi.axhline(y=0,  color=col_truth, ls="--")

    for i, mass_bin in enumerate(dictions[0]):
        ax = axes[i]
        fig = figs[i]
        fig.suptitle(mass_bin + " halos")
        for j, radius_bin in enumerate(dictions[0][mass_bin]):
            axi = ax[j]
            col = cols[j]
            axi.set_title(radius_bin + " radius bin")

            for ii, diction in enumerate(dictions):

                for k1, val1 in enumerate(diction[mass_bin][radius_bin]):
                    if j == 0 and k1 == 0:
                        axi.scatter(k1, round(diction[mass_bin][radius_bin][val1], 3), marker="^", c=col,
                                    label=labels[ii])
                        axi.legend(loc="best")
                    else:
                        axi.scatter(k1, round(diction[mass_bin][radius_bin][val1], 3), marker="^", c=col)

            axi.set_ylim(-1, 1.2)

            axi.set_xticks([0, 1 , 2])
            axi.set_xticklabels(list(dictions[0][mass_bin][radius_bin].keys()))

        plt.savefig("/Users/lls/Documents/deep_halos_files/z99/ics_res75/" + mass_bin + "_halos_comparison.png")
        plt.clf()
