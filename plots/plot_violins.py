import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath("./DeepHalos/"))))
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from plots import predictions_functions as pf


def plot_violin(true, predicted, bins_violin=None,labels=None, return_stats="median", col=None, vert=True, alpha=0.3,
                box=False, figsize=(6.9, 5.2)):
    if len(true) == 2:
        if col is None:
            col1, col1_violin, col2, col2_violin = "#8C4843", "#A0524D", "#406D60", "#55917F"
        else:
            col1, col2 = col
            col1_violin, col2_violin = col

        if bins_violin is None:
            true_all = np.concatenate(true)
            bins_violin = np.linspace(true_all.min(), true_all.max(), 13)

        fig, ax = pf.compare_two_violin_plots(predicted[0], true[0], predicted[1], true[1],
                                              bins_violin, label1=labels[0], label2=labels[1], return_stats=return_stats,
                                              col1=col1, col2=col2, col1_violin=col1_violin, col2_violin=col2_violin,
                                              alpha1=alpha[0], alpha2=alpha[1], figsize=figsize,
                                              edge1='black', edge2='black')
        plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.48, 0.5, 0.5), framealpha=1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        #plt.subplots_adjust(left=0.14)
    else:
        if col is None:
            col1 = "#8C4843",
            col1_violin = "#A0524D"
        else:
            col1 = col
            col1_violin = col

        if bins_violin is None:
            bins_violin = np.linspace(true.min(), true.max() + 0.01, 13, endpoint=True)
        fig, ax = pf.violin_plot(predicted, true, bins_violin, path=None, label1=labels, return_stats=return_stats,
                                 figsize=figsize, title=None, col1=col1, col1_violin=col1_violin, alpha1=alpha,
                                 edge1='black', fig=None, axes=None, vert=vert, box=box)

        plt.subplots_adjust(left=0.13)
    return fig, ax



if __name__ == "__main__":

    path = "/Users/lls/Documents/mlhalos_files/LightGBM/CV_only_reseed_sim/"
    truth = np.load(path + "/truth_shear_den_test_set.npy")
    den_pred = np.load(path + "/pred_density_test_set.npy")

    bins_valid = np.array([11.42095852, 11.62095852, 11.82095852, 12.02095852, 12.22095852, 12.42095852, 12.62095852,
                     12.82095852, 13.02095852, 13.22095852, 13.42095852])

    ids_science = np.where((truth >= bins_valid.min()) & (truth <= bins_valid.max()))[0]
    truth_subset = truth[ids_science]
    den_subset = den_pred[ids_science]

    p1 = np.load("pred1_80.npy")
    t1 = np.load("truth1.npy")
    ids1 = np.where((t1 >= bins_valid.min()) & (t1 <= bins_valid.max()))[0]
    p1_ids = p1[ids1]
    t1_ids = t1[ids1]


    bins_violin = np.linspace(truth_subset.min(), truth_subset.max(), 15)
    # col1="#67a9cf"
    # col2="#ef8a62"
    # col1= color[0]
    # col2= color[1]
    col1 = "#8C4843"
    col2 = "#406D60"
    col1_violin = "#A0524D"
    col2_violin = "#55917F"
    fig, ax = pf.compare_two_violin_plots(den_subset, truth_subset, p1_ids, t1_ids,
                                          bins_violin, label1="GBT", label2="CNN",
                                col1=col1, col2=col2, col1_violin=col1_violin, col2_violin=col2_violin,
                                alpha1=0.3, alpha2=0.3, figsize=(6.9, 5.2),
                                edge1=col1, edge2=col2)
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.48, 0.5, 0.5), framealpha=1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.subplots_adjust(left=0.14)
    # plt.savefig("/Users/lls/Documents/Papers/regression_paper1/violins_den_vs_shear.pdf")