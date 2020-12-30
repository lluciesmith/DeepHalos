import numpy as np
import pynbody
import matplotlib.pyplot as plt


def num_halos(particle_mass, log_bins, m_particle):
    n, b = np.histogram(np.log10(particle_mass[particle_mass > 0]), log_bins)
    mid_b = 10**((log_bins[1:] + log_bins[:-1])/2)
    n_h = n * m_particle/mid_b
    return n_h, mid_b


if __name__ == "__main__":
    path = '/Users/lls/Desktop/data_transfer/'
    ic = pynbody.load(path + "reseed1_simulation/snapshots/IC.gadget3")
    ic.physical_units()

    mass_particle = ic["mass"][0]
    log_bins = np.linspace(10.42, 15, 75)

    def total_halos_sims(sims):
        number_halos = np.zeros((len(sims), len(log_bins)-1))

        for i, sim in enumerate(sims):
            if sim == "0":
                halo_mass_p = np.load(path + "training_simulation/halo_mass_particles.npy")
            else:
                halo_mass_p = np.load(path + "reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")
            num_h, bins = num_halos(halo_mass_p, log_bins, mass_particle)
            number_halos[i] = num_h
        return mid_bins, number_halos

    # Halo mass function
    sims_9 = ["0", "1", "2", "4", "5", "7", "8", "9", "10"]
    mid_bins, number_halos_9 = total_halos_sims(sims_9)
    total_num_halos_9 = np.sum(number_halos_9, axis=0)
    err_9 = np.sqrt(total_num_halos_9)

    all_sims = ["%i" % i for i in np.arange(22)]
    all_sims.remove("3")
    all_sims.remove("6")
    all_sims.append("6")
    mid_bins, number_halos = total_halos_sims(all_sims)
    total_num_halos_20 = np.sum(number_halos, axis=0)
    err_20 = np.sqrt(total_num_halos_20)

    plt.errorbar(mid_bins, total_num_halos_9, yerr=err_9, label="total from 9 simulations", color="k")
    plt.errorbar(mid_bins, total_num_halos_20, yerr=err_20, label="total from 20 simulations", color="red")
    for i in range(len(number_halos)):
        plt.loglog(mid_bins, number_halos[i], alpha=0.3)
    plt.legend(loc="best")
    plt.xlabel("Halo mass")
    plt.ylabel("Number of halos")
    plt.subplots_adjust(bottom=0.14, left=0.15)
    plt.savefig("HMFs.png")

    # Error vs halo mass

    mean_num_halos_per_bin = np.mean(number_halos, axis=0)
    err_one_sim = 1/np.sqrt(mean_num_halos_per_bin)

    tot_rel_err = 1/err

    plt.figure()
    plt.plot(mid_bins[tot_rel_err!=np.inf], tot_rel_err[tot_rel_err != np.inf], color="k",
             label="total from 9 simulations")
    plt.plot(mid_bins[tot_rel_err != np.inf], err_one_sim[tot_rel_err != np.inf] * 1/np.sqrt(20),
             color="red", label="total expected from 20 simulations")
    plt.plot(mid_bins[tot_rel_err != np.inf], err_one_sim[tot_rel_err != np.inf] * 1/np.sqrt(40),
             color="orange", label="total expected from 40 simulations")
    for i in range(len(sims)):
        err_i = 1/np.sqrt(number_halos[i])
        plt.plot(mid_bins[err_i != np.inf], err_i[err_i != np.inf], alpha=0.3)
    plt.ylabel(r"$1/\sqrt{\mathrm{Number \, of \, halos}}$")
    plt.xlabel("Halo mass")

    xmin = plt.ylim()[0]
    plt.axhline(y=0.3, color="grey", label="threshold used in GBT")
    plt.plot([10 ** 14, 10 ** 14], [xmin, 0.3], color="grey", ls="--")
    # plt.plot([3 * 10 ** 14, 3* 10 ** 14], [xmin, 0.3], color="grey", ls="--")

    plt.subplots_adjust(bottom=0.14, left=0.15)
    plt.xscale("log")
    plt.legend(loc="best", fontsize=15)
    plt.savefig("error_vs_halo_mass.png")




    log_bins = np.arange(10.5, 15, 0.1)
    M, sig, N = pynbody.analysis.hmf.halo_mass_function(f, log_M_min=10.5, log_M_max=15)
    N_halos = N * (50**3) * 0.1

    number_halos = np.zeros((len(sims), len(log_bins) - 1))
    for i, sim in enumerate(sims):
        if sim == "0":
            halo_mass_p = np.load(path + "training_simulation/halo_mass_particles.npy")
        else:
            halo_mass_p = np.load(path + "reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")

        halo_mass_p = halo_mass_p * 0.701
        num_h, bins = num_halos(halo_mass_p, log_bins, mass_particle * 0.701)
        number_halos[i] = num_h

    th_error = np.sqrt(N_halos)
    plt.errorbar(M/0.701, N_halos / N_halos, yerr=th_error / N_halos, color="k")
    for i in range(len(sims)):
        plt.plot(M/0.701, number_halos[i] / N_halos, alpha=0.3)

    plt.xscale("log")
    plt.xlabel("Halo mass")

    plt.xlabel("Halo")
