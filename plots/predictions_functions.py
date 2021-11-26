"""
A collection of function useful to handle the predicted log halo mass
form the machine learning algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
import sys
import scipy
import scipy.stats as stats
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from utilss import distinct_colours as dc
from matplotlib.ticker import FormatStrFormatter


def get_confusion_matrix_per_halo_mass(predictions, truth, epsilon=0.5):
    halo_masses_unique = np.unique(truth)

    TP = []
    FN = []
    FP = []
    TN = []

    for h in halo_masses_unique:
        ids_h = np.where(truth == h)[0]
        ids_not_h = np.where(truth != h)[0]

        dist_epsilon = h + epsilon
        TP.append(len(np.where(predictions[ids_h] <= dist_epsilon)[0]))
        FN.append(len(np.where(predictions[ids_h] > dist_epsilon)[0]))
        FP.append(len(np.where(predictions[ids_not_h] <= dist_epsilon)[0]))
        TN.append(len(np.where(predictions[ids_not_h] > dist_epsilon)[0]))

    TPR = np.array(TP)/(np.array(TP) + np.array(FN))
    FPR = np.array(FP)/(np.array(FP) + np.array(TN))
    return TPR, FPR


def get_confusion_matrix_bins_halo_mass(predictions, truth, num_bins=50):
    ncount, bins = np.histogram(truth, bins=num_bins)
    mid_bins = (bins[1:] + bins[:-1])/2

    TP = []
    FN = []
    FP = []
    TN = []

    for i in range(len(bins) - 1):
        in_bin = (truth >= bins[i]) & (truth <= bins[i+1])
        ids_h = np.where(in_bin)[0]
        print(len(ids_h))

        if len(ids_h) == 0:
            TP.append(0)
            FN.append(0)
            FP.append(0)
            TN.append(0)
        else:
            out_bin = (truth < bins[i]) | (truth > bins[i+1])
            ids_not_h = np.where(out_bin)[0]
            print(len(ids_not_h))

            h = mid_bins[i]
            dist_epsilon = h + epsilon

            TP.append(len(np.where(predictions[ids_h] <= dist_epsilon)[0]))
            FN.append(len(np.where(predictions[ids_h] > dist_epsilon)[0]))
            FP.append(len(np.where(predictions[ids_not_h] <= dist_epsilon)[0]))
            TN.append(len(np.where(predictions[ids_not_h] > dist_epsilon)[0]))

    ignore_0_bins = np.where(np.array(FP) != 0)[0]

    TPR = np.array(TP)[ignore_0_bins]/(np.array(TP)[ignore_0_bins] + np.array(FN)[ignore_0_bins])
    FPR = np.array(FP)[ignore_0_bins]/(np.array(FP)[ignore_0_bins] + np.array(TN)[ignore_0_bins])
    return TPR, FPR, mid_bins[ignore_0_bins]


def get_bias_and_variance_prediction_each_bin(predicted, truth, bins):

    b = []
    v = []

    for i in range(len(bins) - 1):
        ids = (truth >= bins[i]) & (truth <= bins[i + 1])

        b.append(np.mean(truth[ids]) - np.mean(predicted[ids]))
        v.append(np.var(predicted[ids]))
    return np.array(b), np.array(v)

def get_summary_statistic_each_bin(predicted, truth, bins, stats="mean"):
    n = []
    m = []
    v = []
    s = []
    k = []

    for i in range(len(bins) - 1):
        ids = (truth >= bins[i]) & (truth <= bins[i + 1])
        num, minmax, mean, var, skew, kurt = scipy.stats.describe(predicted[ids])
        n.append(num)
        m.append(mean)
        v.append(var)
        s.append(skew)
        k.append(scipy.stats.kurtosis(predicted[ids], fisher=False))

    return np.array(n), np.array(m), np.array(v), np.array(s), np.array(k)


def histogram_predictions_in_bins(pred, truth, bins, label=" ", save=False, path=" "):
    for i in range(len(bins)-1):
        ids = (truth >= bins[i]) & (truth <= bins[i + 1])

        mae_i = mae(truth[ids], pred[ids])

        plt.figure()
        plt.hist(pred[ids], bins=50, label=label + "\n (mae = %.3f)" % mae_i, histtype="step", density=True)

        plt.axvline(x=bins[i], color="k")
        plt.axvline(x=bins[i + 1], color="k")
        plt.xlim(10, 15)
        plt.xlabel("Predicted log masses")
        plt.legend(loc="best")
        plt.subplots_adjust(bottom=0.15)
        if save is True:
            plt.savefig(path + "bin_" + str(i) + "_" + label + ".png")


def compare_histogram_predictions_each_bin(pred1, pred2, truth, bins, label1=" ", label2=" ", save=False, path=" "):
    for i in range(len(bins)-1):
        ids1 = (truth >= bins[i]) & (truth <= bins[i + 1])
        ids2 = (truth >= bins[i]) & (truth <= bins[i + 1])

        mae1 = mae(truth[ids1], pred1[ids1])
        mae2 = mae(truth[ids2], pred2[ids2])

        plt.figure()
        plt.hist(pred1[ids1], bins=50, label=label1 + "\n (mae = %.3f)" % mae1, histtype="step", density=True)
        plt.hist(pred2[ids2], bins=50, label=label2 + "\n (mae = %.3f)" % mae2, histtype="step", density=True)

        plt.axvline(x=bins[i], color="k")
        plt.axvline(x=bins[i + 1], color="k")
        plt.xlim(10, 15)
        plt.xlabel("Predicted log masses")
        plt.legend(loc="best")
        plt.subplots_adjust(bottom=0.15)
        if save is True:
            plt.savefig(path + "bin_" + str(i) + "_" + label1 + "_vs_" + label2 + ".png")


def get_distributions_for_violin_plots(predicted1, true1, predicted2, true2, bins, return_stats="median"):
    distr_pred1, distr_mean1 = get_predicted_masses_in_each_true_m_bin(bins, predicted1, true1,
                                                                      return_stats=return_stats)
    distr_pred2, distr_mean2 = get_predicted_masses_in_each_true_m_bin(bins, predicted2, true2,
                                                                          return_stats=return_stats)
    return distr_pred1, distr_mean1, distr_pred2, distr_mean2


def get_median_true_distribution_in_bins(truth, bins, return_stats="median"):
    median_truth_each_bin = []

    for i in range(len(bins) - 1):
        indices_each_bin = np.where((truth >= bins[i]) & (truth < bins[i + 1]))[0]
        indices_each_bin = indices_each_bin.astype("int")

        truth_each_bin = truth[indices_each_bin]
        if return_stats == "median":
            truth_mean = np.median(truth_each_bin)
        else:
            truth_mean = np.mean(truth_each_bin)

        median_truth_each_bin.append(truth_mean)

    median_truth_each_bin = np.array(median_truth_each_bin)
    return median_truth_each_bin


def sns_violin_plot(predicted, truth, bins,
                path=None, label1="distribution1", return_stats="median", figsize=(8, 6),
                title=None, col1=None, col1_violin=None, alpha1=1, edge1='black',
                fig=None, axes=None, vert=True, box=False):

    bins_mid = (bins[1:] + bins[:-1]) / 2
    data = {"truth": truth,
            r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$": predicted}

    df = pd.DataFrame(data=data)
    df[r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$"] = pd.cut(x=df['truth'], bins = bins, labels = bins_mid)

    new_textsize = figsize[0] / (6.9 / 17)
    new_ticksize = figsize[0] / (6.9 / 18)
    mpl.rcParams['ytick.labelsize'] = new_ticksize
    mpl.rcParams['xtick.labelsize'] = new_ticksize

    if fig is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    vplot1 = sns.violinplot(x=r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$",
                            y=r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", data=df, ax=axes,
                            scale="width", inner="box", color=dc.get_distinct(4)[1])
    axes.plot(np.arange(len(bins)), bins, color="k")
    axes.get_xticklabels(['%.2f' % x for x in bins_mid])

    # if vert is True:
    #     if return_stats is not None:
    #         axes.errorbar(xaxis, distr_mean1, xerr=width_xbins / 2, color=col1, fmt="o", label=label1)
    #     # axes.set_xlim(bins.min() - 0.01, bins.max() + 0.01)
    #     axes.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$")
    #     axes.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$")
    # else:
    #     if return_stats is not None:
    #         axes.scatter(distr_mean1, xaxis, s=5, color=col1, label=label1)
    #     # axes.set_ylim(bins.min() - 0.01, bins.max() + 0.01)
    #     axes.set_ylabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$")
    #     axes.set_xlabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$")

    # if label1 is not None:
    #     axes.legend(loc="best", fontsize=new_textsize, framealpha=1.)
    #
    # plt.subplots_adjust(bottom=0.14, left=0.1)
    # if title is not None:
    #     plt.title(title)
    #     plt.subplots_adjust(top=0.9)
    #
    # if path is not None:
    #     plt.savefig(path)

    return fig, axes


def violin_plot(predicted, truth, bins,
                path=None, label1="distribution1", return_stats="median", figsize=(8, 6),
                title=None, col1=None, col1_violin=None, alpha1=1, edge1='black',
                fig=None, axes=None, vert=True, box=False):

    distr_pred1, distr_mean1 = get_predicted_masses_in_each_true_m_bin(bins, predicted, truth,
                                                                       return_stats=return_stats)
    print(distr_mean1)

    new_textsize = figsize[0] / (6.9 / 17)
    new_ticksize = figsize[0] / (6.9 / 18)
    mpl.rcParams['ytick.labelsize'] = new_ticksize
    mpl.rcParams['xtick.labelsize'] = new_ticksize

    width_xbins = np.diff(bins) * 0.95
    xaxis = (bins[:-1] + bins[1:]) / 2

    if col1 is None:
        col1 = "b"
        if col1_violin is None:
            col1_violin = col1

    if fig is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    vplot1 = axes.violinplot(distr_pred1, positions=xaxis, widths=width_xbins, showextrema=False, showmeans=False,
                             showmedians=False, vert=vert)
    for b in vplot1["bodies"]:
        b.set_facecolor(col1_violin)
        b.set_edgecolor(edge1)
        b.set_alpha(alpha1)
        b.set_linewidths(1)

    if box is True:
        medianprops = {'color': 'white', 'linewidth': 1}
        boxprops = {'color': 'black', 'linestyle': '-'}
        whiskerprops = {'color': 'black', 'linestyle': '-'}

        box1 = axes.boxplot(distr_pred1, positions=xaxis, sym='', widths=width_xbins/10, vert=vert,
                            showfliers=False, medianprops=medianprops, boxprops=boxprops,
                            whiskerprops=whiskerprops, patch_artist=True)
        for patch in box1['boxes']:
            patch.set_facecolor("k")

    axes.plot(bins, bins, color="k")
    axes.set_xticks(np.linspace(axes.get_xlim()[0] + 0.1, axes.get_xlim()[1] - 0.1, 6))
    axes.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if vert is True:
        if return_stats is not None:
            axes.errorbar(xaxis, distr_mean1, xerr=width_xbins / 2, color=col1, fmt="o", label=label1)
        axes.set_xlim(bins.min() - 0.01, bins.max() + 0.01)
        axes.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", fontsize=20)
        axes.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", fontsize=20)
    else:
        if return_stats is not None:
            axes.scatter(distr_mean1, xaxis, s=5, color=col1, label=label1)
        axes.set_ylim(bins.min() - 0.01, bins.max() + 0.01)
        axes.set_ylabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$", fontsize=20)
        axes.set_xlabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$", fontsize=20)

    if label1 is not None:
        axes.legend(loc="best", fontsize=new_textsize, framealpha=1.)

    plt.subplots_adjust(bottom=0.14, left=0.1)
    if title is not None:
        plt.title(title)
        plt.subplots_adjust(top=0.9)

    if path is not None:
        plt.savefig(path)

    return fig, axes


def two_violin_plot(distr_pred1, distr_mean1, distr_pred2, distr_mean2, bins, truth, path=None, label1="distribution1",
                    label2="distribution2", title=None, col1=None, col1_violin=None, col2=None, col2_violin=None,
                    figsize=(8, 6), return_stats=None,
                    alpha1=1, edge1='black', alpha2=1, edge2='black'):


#    new_textsize = figsize[0] / (6.9 / 17)
#    new_labelsize = figsize[0] / (6.9 / 22)
#    new_ticksize = figsize[0] / (6.9 / 18)
#    mpl.rcParams['ytick.labelsize'] = new_ticksize
#    mpl.rcParams['xtick.labelsize'] = new_ticksize

    width_xbins = np.diff(bins)
    xaxis = (bins[:-1] + bins[1:]) / 2
    xaxis_median = get_median_true_distribution_in_bins(truth, bins, return_stats="median")

    if col1 is None:
        col1 = "b"
        if col1_violin is None:
            col1_violin = col1
    if col2 is None:
        col2 = "r"
        if col2_violin is None:
            col2_violin = col2

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    vplot1 = axes.violinplot(distr_pred1, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    for b in vplot1["bodies"]:
        b.set_facecolor(col1_violin)
        b.set_edgecolor(edge1)
        b.set_alpha(alpha1)
        b.set_linewidths(1)

    vplot = axes.violinplot(distr_pred2, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    for b in vplot["bodies"]:
        b.set_facecolor(col2_violin)
        b.set_edgecolor(edge2)
        b.set_alpha(alpha2)
        b.set_linewidths(1)
    
    if return_stats is not None:
        axes.errorbar(xaxis, distr_mean1, xerr=width_xbins / 2, color=col1, fmt="o", label=label1)
        axes.errorbar(xaxis, distr_mean2, xerr=width_xbins / 2, color=col2, fmt="o", label=label2)

    axes.plot(bins, bins, color="k")
    axes.set_xlim(bins.min() - 0.01, bins.max() + 0.01)
    # axes.set_ylim(bins.min() - 0.1, bins.max() + 0.1)

    # axes.set_xlim(10.5, 15)
    # axes.set_ylim(axes.xlim())

    axes.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$")
    axes.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$")
    axes.legend(loc="best", framealpha=1.)
    plt.subplots_adjust(bottom=0.14, left=0.1)
    if title is not None:
        plt.title(title)
        plt.subplots_adjust(top=0.9)

    if path is not None:
        plt.savefig(path)

    return fig, axes


def compare_two_violin_plots(predicted1, true1, predicted2, true2, bins, path=None,
                             label1="distribution1", label2="distribution2",
                             return_stats="median", title=None, col1=None, col2=None, col1_violin=None,
                             col2_violin=None,
                             figsize=(8, 6),
                             alpha1=1, edge1='black', alpha2=1, edge2='black'):
    distr_pred1, distr_mean1, distr_pred2, distr_mean2 = get_distributions_for_violin_plots(predicted1, true1,
                                                                                            predicted2, true2, bins,
                                                                                            return_stats=return_stats)
    #assert np.allclose(true1, true2)
    f, ax = two_violin_plot(distr_pred1, distr_mean1, distr_pred2, distr_mean2, bins, true1, path=path, label1=label1,
                            label2=label2, title=title, col1=col1, col1_violin=col1_violin, col2=col2,
                            col2_violin=col2_violin, figsize=figsize, alpha1=alpha1, edge1=edge1, alpha2=alpha2,
                            edge2=edge2, return_stats=return_stats)
    return f, ax


def get_predicted_masses_in_each_true_m_bin(bins, mass_predicted_particles, true_mass_particles,
                                            return_stats="median"):
    log_pred_bins = []
    mean_each_bin = []

    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            print(i)
            indices_each_bin = np.where((true_mass_particles >= bins[i]) & (true_mass_particles <= bins[i + 1]))[0]
        else:
            indices_each_bin = np.where((true_mass_particles >= bins[i]) & (true_mass_particles < bins[i + 1]))[0]
        indices_each_bin = indices_each_bin.astype("int")

        predicted_each_bin = mass_predicted_particles[indices_each_bin]
        if return_stats == "median":
            predicted_mean = np.median(predicted_each_bin)

        elif return_stats == "mode":
            n, b = np.histogram(predicted_each_bin, bins=80)
            mid_bins = (b[1:] + b[:-1])/2
            predicted_mean = float(mid_bins[n == n.max()].max())
            print(predicted_mean)
        else:
            predicted_mean = np.mean(predicted_each_bin)

        log_pred_bins.append(list(predicted_each_bin))
        mean_each_bin.append(predicted_mean)

    mean_each_bin = np.array(mean_each_bin)
    return log_pred_bins, mean_each_bin


