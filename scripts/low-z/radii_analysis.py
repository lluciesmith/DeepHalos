import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from mlhalos import parameters
from mlhalos import distinct_colours
import numpy as np
import matplotlib.pyplot as plt
from utils import radius_functions_deep as rf


if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(initial_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/IC.gadget3",
                                                final_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/snapshot_099")

    ids_i = np.loadtxt("/Users/lls/Documents/mlhalos_files/reseed50/CNN_results/reseed_1_random_training_set.txt")
    ids_i = ids_i.astype("int")

    r_i = np.load("/Users/lls/Documents/mlhalos_files/reseed50/radii_properties_ids_random_training_set_above_1e11.npy")

    ti = np.load("/Users/lls/Documents/deep_halos_files/z0/correct_ordering/truth1_80.npy")

    ids = ids_i[np.where(ti >= 12)[0]]
    halo_ids_i = ic.final_snapshot[ids_i]['grp']
    ti_above_1e11 = ti[halo_ids_i <= 5300]
    r = r_i[ti_above_1e11 >= 12]

    # p1 = np.load("/Users/lls/Documents/deep_halos_files/z0/correct_ordering/pred1_80.npy")
    # t1 = np.load("/Users/lls/Documents/deep_halos_files/z0/high_mass/true1_60.npy")
    # p1 = np.load("/Users/lls/Documents/deep_halos_files/z0/high_mass/predicted1_60.npy")

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

    #### PLOT #####

    color = distinct_colours.get_distinct(4)
    c0 = color[0]
    c1 = color[1]
    c2 = color[3]
    lw = 1.8
    density = False

    m_p = ic.initial_conditions['mass'][0]
    m_min = m_min = np.log10(m_p * 300)

    mb = [11, 12, 13, truth_inner.max()]
    b = np.linspace(-4, 4, 50)
    rf.plot_diff_predicted_true_radial_ranges(pred_inner, truth_inner, den_mid, truth_mid, den_outer, truth_outer,
                                              b, mb, c0, c1, c2, i1, m0, m1, o0,
                                              lw=lw, density=density, figsize=(12, 5.2), fontsize=15)
    plt.yscale("log")
    plt.subplots_adjust(top=0.93)
    # plt.savefig("/Users/lls/Documents/deep_halos_files/z0/predictions_mass_and_radius_bins.png")
    plt.savefig("/Users/lls/Documents/deep_halos_files/z0/high_mass/predictions_mass_and_radius_bins.png")


    def mean_err(true, pred, bins=20):
        t_bins = np.linspace(m_min, 12 , bins, endpoint=True)
        res = (pred - true)
        mean = []
        std = []
        for i in range(len(t_bins) - 1):
            ind = np.where((true >= t_bins[i]) & (true <= t_bins[i + 1]))[0]
            mean.append(np.mean(res[ind]))
            std.append(np.std(res[ind]))

        return (t_bins[1:] + t_bins[:-1])/2, np.array(mean), np.array(std)


    t = np.load("/Users/lls/Documents/deep_halos_files/z0/high_mass/true1_60.npy")
    p = np.load("/Users/lls/Documents/deep_halos_files/z0/high_mass/predicted1_60.npy")
    b, m, s = mean_err(t, p, 10)

    fig, ax = plt.subplots()
    ind = (t >= m_min) & (t <= 12)
    # ind = (t<=12)
    err = p - t

    ax.scatter(t[ind], err[ind], s=0.5)
    ax.axhline(y=0, color="grey")
    ax.errorbar(b, m, yerr=s, color="k")
    ax.axhline(y=0.044, color="C1", label=r"std in bin $12 \leq \log(M)\leq 13$")
    plt.legend(loc="best")

    ax.set_xlabel(r"$\log(M_\mathrm{true})$")
    ax.set_ylabel(r"$\log(M_\mathrm{predicted}/M_\mathrm{true})$")

    def halomass_to_pnumber(x):
        m_p = ic.initial_conditions['mass'][0]
        V = 10**x / m_p
        return ["%i" % z for z in V]

    def pnumber_to_halomass(x):
        m_p = ic.initial_conditions['mass'][0]
        V = m_p * x
        return ["%.3f" % z for z in V]

    ax2 = ax.twiny()
    xticks = ax.get_xticks()
    xticks_new = xticks[1:-1]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xticks_new)
    ax2.set_xticklabels(halomass_to_pnumber(xticks_new))
    ax2.set_xlabel("Number of particles")
    plt.subplots_adjust(bottom=0.14, top=0.87)
