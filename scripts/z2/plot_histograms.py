import numpy as np
import matplotlib.pyplot as plt


def plot_histogram_predictions_in_bins(pred, true, blow, bhigh, bins, loc_text=None, radius_bins=False,
                                       r_properties=[],
                                       density=False):
    p_mid = pred[(true >= blow) & (true <= bhigh)]
    r_mid = r_properties[(true >= blow) & (true <= bhigh)]
    f, ax = plt.subplots()

    if radius_bins is False:
        n_mid, b, p = ax.hist(p_mid, bins=bins, histtype="step", lw=2)
        n_t, b, p = ax.hist(true[(true >= blow) & (true <= bhigh)], bins=bins, histtype="step", ls="--", color="k")

        n = np.sum(p_mid[(p_mid < blow) | (p_mid > bhigh)]) / np.sum(p_mid) * 100

        ymax = plt.ylim()[1]
        if loc_text is None:
            loc_text = np.zeros(2, )
            loc_text[1] = 0.7 * ymax
            print(blow)
            if blow < 12:
                loc_text[0] = 12.5
            else:
                print("true")
                loc_text[0] = 10

        print(loc_text)
        ax.text(loc_text[0], loc_text[1],
                "%.1f" % n + "$\%$" + " particles outside\n %.1f $\leq\log(M/M_\odot)\leq$ %.1f " % (blow, bhigh),
                fontsize=14)
        ax.set_title(r"z=0 sub-boxes, $%.1f\leq\log(M_\mathrm{true}/M_\odot)\leq %.1f$" % (blow, bhigh))

    else:
        # n_in1, b, p = ax.hist(p_mid[r_mid <= 0.3], bins=bins, histtype="step", lw=1.5, label="inner", density=True)
        # n_mid1, b, p = ax.hist(p_mid[(r_mid > 0.5) & (r_mid <= 0.8)], bins=bins, histtype="step",
        #                       lw=1.5, label="mid", density=True)
        # n_out1, b, p = ax.hist(p_mid[r_mid >= 1], bins=bins, histtype="step", lw=1.5, label="outer", density=True)
        # plt.legend(loc="best")
        n_in1, b, p = ax.hist(p_mid[r_mid <= 0.3], bins=bins, histtype="step", lw=1.5,
                              label=r"$r/r_\mathrm{vir} \leq 0.3$", density=density)
        n_out1, b, p = ax.hist(p_mid[r_mid >= 0.8], bins=bins, histtype="step", lw=1.5,
                               label=r"$r/r_\mathrm{vir} > 0.8$", density=density)
        plt.legend(loc="best")

    ax.axvline(x=bhigh, color="k", ls="--")
    ax.axvline(x=blow, color="k", ls="--")
    ax.set_xlabel(r"$\log(M_\mathrm{predicted}/M_\odot)$")
    ax.set_ylabel("$N_\mathrm{particles}$")
    plt.subplots_adjust(top=0.92, bottom=0.14)
    return f

if __name__ == "__main__":
    bins = np.arange(10, 14.5, step=0.15)

    ids_tested = np.loadtxt("/Users/lls/Documents/deep_halos_files/regression/z0/training_sim_random_training_set.txt",
                            dtype="int")
    r_prop_all_ids_in_halos = np.load(
        "/Users/lls/Documents/mlhalos_files/correct_radii_properties_all_ids_in_halos_upto_2436.npy")
    ids_in_halo = r_prop_all_ids_in_halos[:,0].astype("int")
    r_prop_ids_tested = [r_prop_all_ids_in_halos[ids_in_halo == i] for i in ids_tested]
    r_prop_ids_tested = np.array(r_prop_ids_tested).reshape(20000, 3)

    path = "/Users/lls/Documents/deep_halos_files/regression/mixed_sims/standardize_wdropout/"
    t = np.load(path + "truth0.npy")
    p0 = np.load(path + "pred0_80.npy")

    mass_bins = np.linspace(t.min(), t.max(), 15)
    for i in range(len(mass_bins) - 1):
        fi = plot_histogram_predictions_in_bins(p0, t, mass_bins[i], mass_bins[i + 1], bins, radius_bins=True,
                                                r_properties=r_prop_ids_tested[:, 2], density=True)
        plt.savefig("/Users/lls/Documents/deep_halos_files/regression/mixed_sims/standardize_wdropout/histograms/hist_" + "%.1f" % mass_bins[i]
                    + "_" + "%.1f" % mass_bins[i + 1] + ".png")
        # plt.savefig("/Users/lls/Documents/deep_halos_files/regression/z0/histograms/radii/hist_" + "%.1f" %
        # mass_bins[i]  + "_" + "%.1f" % mass_bins[i + 1] + ".png")
        plt.clf()

