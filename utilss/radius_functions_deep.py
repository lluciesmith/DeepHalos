import sys
sys.path.append('/Users/lls/Documents/Projects/DeepHalos')
from utilss import radius_functions_deep as rf
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from mlhalos import distinct_colours
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
                                      figsize=(10, 5.2), col_truth="dimgrey", lw=1.8,
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


def get_indices_array_within_bounds(array,  bin_lower_lim=0, bin_upper_lim=100):
    return np.where((array >= bin_lower_lim) & (array <= bin_upper_lim))[0]


def plot_radial_cat_in_mass_bins(predictions, truth, radii, inner=True, mid=True, out=True,
                                 inner_lim=(0.0000000001, 0.2), mid_lim=(0.4, 0.6), outer_lim=(0.8, 100),
                                 bins_r=None, bins_m=None, num_bins=50, errorbars=True, legend=True, col=None,
                                 figsize=(10, 5.1), col_truth="dimgrey", lw=1.8, density=True, fig=None, axes=None):
    ind_inner = get_indices_array_within_bounds(radii, bin_lower_lim=inner_lim[0], bin_upper_lim=inner_lim[1])
    ind_mid = get_indices_array_within_bounds(radii, bin_lower_lim=mid_lim[0], bin_upper_lim=mid_lim[1])
    ind_outer = get_indices_array_within_bounds(radii, bin_lower_lim=outer_lim[0], bin_upper_lim=outer_lim[1])

    pred_inner, truth_inner = predictions[ind_inner], truth[ind_inner]
    pred_mid, truth_mid = predictions[ind_mid], truth[ind_mid]
    pred_outer, truth_outer = predictions[ind_outer], truth[ind_outer]

    if bins_m is None:
        bins_m = [11, 12, 13, truth_inner.max()]
    if bins_r is None:
        max_b = max(abs(predictions - truth)) + 0.1
        bins_r = np.linspace(-max_b, max_b, num_bins)
    if col is None:
        color = distinct_colours.get_distinct(4)
    else:
        color=[col, col, col, col]
    f, axes = plot_diff_predicted_true_radial_ranges(pred_inner, truth_inner,
                                                     pred_mid, truth_mid,
                                                     pred_outer,  truth_outer,
                                                     bins_r, bins_m, color[0], color[1], color[3], inner_lim[1],
                                                     mid_lim[0], mid_lim[1], outer_lim[0],
                                                     inner=inner, mid=mid, out=out, legend=legend,
                                                     figsize=figsize, col_truth=col_truth, lw=lw, density=density,
                                                     fig=fig, axes=axes, errorbars=errorbars)
    return f, axes


def plot_diff_predicted_true_mass_ranges(predictions, truths, mass_bins, xbins,
                                           figsize=(10, 5.1),
                                           col_truth="dimgrey", lw=1.8,
                                          density=True, fig=None, axes=None, color="C0", label=None):
    if fig is None:
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharex=True)
    else:
        f=fig
        (ax1, ax2, ax3) = axes

    ax1.axvline(x=0, color=col_truth, ls="--")
    ax2.axvline(x=0, color=col_truth, ls="--")
    ax3.axvline(x=0, color=col_truth, ls="--")

    pred_low = (truths >= mass_bins[0]) & (truths < mass_bins[1])
    pred_mid = (truths >= mass_bins[1]) & (truths < mass_bins[2])
    pred_high = (truths >= mass_bins[2]) & (truths < mass_bins[3])

    aa = ax1.hist(predictions[pred_low] - truths[pred_low], bins=xbins,
                 histtype="step", density=density, lw=lw, color=color, label=label)
    ax1.set_title(r"$ %.1f \leq \log(M_{\mathrm{true}}) \leq %.1f$" % (mass_bins[0], mass_bins[1]), fontsize=16)

    _ = ax2.hist(predictions[pred_mid] - truths[pred_mid], bins=xbins,
                 histtype="step", density=density, lw=lw, color=color)
    ax2.set_title(r"$%.1f \leq \log(M_{\mathrm{true}}) \leq %.1f$" % (mass_bins[1], mass_bins[2]), fontsize=16)

    dd = ax3.hist(predictions[pred_high] - truths[pred_high], bins=xbins,
                 histtype="step", density=density, lw=lw, color=color)
    ax3.set_title(r"$%.1f \leq \log(M_{\mathrm{true}}) \leq %.1f$" % (mass_bins[2], mass_bins[3]), fontsize=16)

    plt.subplots_adjust(wspace=0, bottom=0.15, left=0.08, top=0.93)

    if label is not None:
        handles, labels = ax1.get_legend_handles_labels()
        f.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.12, 0.45, 0.5, 0.5), framealpha=1, fontsize=16)
        #ax1.legend(fontsize=16, loc='upper center', bbox_to_anchor=(1., 0.45, 0.5, 0.5), framealpha=1)
    ax1.set_ylabel(r"$n_{\mathrm{particles}}$")
    ax1.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax2.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax3.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    # ax3.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    return f, (ax1, ax2, ax3)


def plot_diff_predicted_true_radial_ranges(den_inner, truth_inner, den_mid, truth_mid, den_outer, truth_outer,
                                          bins_radial_distributions, mass_bins,
                                          col_in, col_mid, col_out, i1, m0, m1, o0, figsize=(10, 5.1),
                                           col_truth="dimgrey", lw=1.8,
                                           inner=True, mid=True, out=True, legend=True,
                                          density=True, fig=None, axes=None, errorbars=True):
    if fig is None:
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharex=True)
    else:
        f = fig
        (ax1, ax2, ax3) = axes

    ax1.axvline(x=0, color=col_truth, ls="--")
    ax2.axvline(x=0, color=col_truth, ls="--")
    ax3.axvline(x=0, color=col_truth, ls="--")

    low_mass_inner = (truth_inner >= mass_bins[0]) & (truth_inner < mass_bins[1])
    low_mass_mid = (truth_mid >= mass_bins[0]) & (truth_mid < mass_bins[1])
    low_mass_outer = (truth_outer >= mass_bins[0]) & (truth_outer < mass_bins[1])

    colors = [col_in, col_mid, col_out]
    if inner is True:
        _0 = ax1.hist(den_inner[low_mass_inner] - truth_inner[low_mass_inner], bins=bins_radial_distributions,
                     histtype="step", density=density, lw=lw, color=col_in)
    if mid is True:
        _1 = ax1.hist(den_mid[low_mass_mid] - truth_mid[low_mass_mid], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_mid)
    if out is True:
        _2 = ax1.hist(den_outer[low_mass_outer] - truth_outer[low_mass_outer], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_out)
    if errorbars is True:
        for _ in [_0, _1, _2]:
            x = (_[1][:-1] + _[1][1:])/ 2.
            ax1.errorbar(x, _[0], yerr=np.sqrt(_[0]), linestyle='None')

    ax1.set_title(r"$ %.2f \leq \log(M_{\mathrm{true}}) \leq %.2f$" % (mass_bins[0], mass_bins[1]), fontsize=16)

    mid_mass_inner = (truth_inner >= mass_bins[1]) & (truth_inner < mass_bins[2])
    mid_mass_mid = (truth_mid >= mass_bins[1]) & (truth_mid < mass_bins[2])
    mid_mass_outer = (truth_outer >= mass_bins[1]) & (truth_outer < mass_bins[2])
    if inner is True:
        __0 = ax2.hist(den_inner[mid_mass_inner] - truth_inner[mid_mass_inner], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_in)
    if mid is True:
        __1 = ax2.hist(den_mid[mid_mass_mid] - truth_mid[mid_mass_mid], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_mid)
    if out is True:
        __2 = ax2.hist(den_outer[mid_mass_outer] - truth_outer[mid_mass_outer], bins=bins_radial_distributions,
                 histtype="step", density=density, lw=lw, color=col_out)
    if errorbars is True:
        for _ in [__0, __1, __2]:
            x = (_[1][:-1] + _[1][1:]) / 2.
            ax2.errorbar(x, _[0], yerr=np.sqrt(_[0]), linestyle='None')

    ax2.set_title(r"$%.2f \leq \log(M_{\mathrm{true}}) \leq %.2f$" % (mass_bins[1], mass_bins[2]), fontsize=16)

    high_mass_inner = (truth_inner >= mass_bins[2]) & (truth_inner <= mass_bins[3])
    high_mass_mid = (truth_mid >= mass_bins[2]) & (truth_mid <= mass_bins[3])
    high_mass_outer = (truth_outer >= mass_bins[2]) & (truth_outer <= mass_bins[3])
    if inner is True:
        aa = ax3.hist(den_inner[high_mass_inner] - truth_inner[high_mass_inner], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_in)
    if mid is True:
        bb = ax3.hist(den_mid[high_mass_mid] - truth_mid[high_mass_mid], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_mid)
    if out is True:
        dd = ax3.hist(den_outer[high_mass_outer] - truth_outer[high_mass_outer], bins=bins_radial_distributions,
                  histtype="step", density=density, lw=lw, color=col_out)
    # print("Stats for outskirts particles in high mass halos:\r\n")
    # print(stats.describe(den_outer[high_mass_outer] - truth_outer[high_mass_outer]))

    if errorbars is True:
        for _ in [aa, bb, dd]:
            x = (_[1][:-1] + _[1][1:]) / 2.
            ax3.errorbar(x, _[0], yerr=np.sqrt(_[0]), linestyle='None')

    ax3.set_title(r"$%.2f \leq \log(M_{\mathrm{true}}) \leq %.2f$" % (mass_bins[2], mass_bins[3]), fontsize=16)

    plt.subplots_adjust(wspace=0, bottom=0.15, left=0.08, top=0.93)

    ax1.set_ylabel(r"$n_{\mathrm{particles}}$")
    ax1.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax2.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax3.set_xlabel(r"$\log(M_{\mathrm{predicted}}/M_{\mathrm{true}})$")
    ax3.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax2.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax1.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    if legend is True:
        plt.figlegend((aa[2][0], bb[2][0], dd[2][0]),
                      (r"$r/\mathrm{r_{vir}} \leq %.1f $" % i1,
                       r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (m0, m1),
                       # r"$%.1f\leq r/\mathrm{r_{vir}}\leq %.1f $" % (o0, o1),
                       r"$r/\mathrm{r_{vir}} > %.1f $" % o0,
                       ),
                       loc='upper center', bbox_to_anchor=(0.12, 0.45, 0.5, 0.5), framealpha=1, fontsize=14)

    # ax1.text(0.2, 0.8, "Low-mass \n haloes", horizontalalignment='center', verticalalignment='center',
    #          fontweight='bold', transform=ax1.transAxes)
    # ax2.text(0.8, 0.8, "Mid-mass \n haloes", horizontalalignment='center', verticalalignment='center',
    #          fontweight='bold', transform=ax2.transAxes)
    # ax3.text(0.8, 0.8, "High-mass \n haloes", horizontalalignment='center', verticalalignment='center',
    #          fontweight='bold', transform=ax3.transAxes)
    return f, (ax1, ax2, ax3)


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
