import numpy as np
import pynbody
import matplotlib.pyplot as plt

ic = pynbody.load("/Users/lls/Desktop/data_transfer/reseed6_simulation/snapshots/IC.gadget2")
ic.physical_units()

f = pynbody.load("/Users/lls/Documents/deep_halos_files/simulations/reseed6_simulation/snapshot_007")
f.physical_units()
b = ic.bridge(f)

h = f.halos()
mass_each_halo = np.load("/Users/lls/Documents/deep_halos_files/simulations/reseed6_simulation/mass_Msol_each_halo_sim_6.npy")

p = np.load("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/seed_20/predicted_sim_6_epoch_09.npy")
t = np.load("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/seed_20/true_sim_6_epoch_09.npy")
ids = np.load("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/ids_larger_validation_set.npy")

r = np.load("/Users/lls/Documents/mlhalos_files/reseed6/reseed6radius_in_halo_particles.npy")[ids]
r_vir = np.load("/Users/lls/Documents/mlhalos_files/reseed6/reseed6_virial_radius_particles.npy")[ids]

# Consider outskirt particles (r/r_vir > 0.8) in high mass halos (log M >13)

try:
    r_right = np.load("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/seed_20"
                      "/r_right.npy")
    r_wrong = np.load("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/seed_20/r_wrong.npy")

except IOError:
    ind_wrong = (r / r_vir > 0.8) & (t >= 13) & (p - t <= -1)
    truths_wrong = t[ind_wrong]
    ids_wrong = ids[ind_wrong]

    r_wrong = []
    for i in range(len(truths_wrong)):
        h_id = int(np.where(np.log10(mass_each_halo) == min(np.log10(mass_each_halo), key=lambda x:abs(x-truths_wrong[i])))[0])
        pynbody.analysis.halo.center(ic[ids_wrong[i]], vel=False, wrap=True)
        prog_particles = b(h[h_id])
        r_wrong.append(list(prog_particles['r']))
    np.save("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/seed_20/r_wrong.npy",
            np.concatenate(r_wrong))

    ind_right = (r / r_vir > 0.8) & (t >= 13) & (p - t > -1)
    truths_right = t[ind_right]
    ids_right= ids[ind_right]

    sub_sample = np.random.choice(range(len(truths_right)), 400)

    r_right = []
    for i in sub_sample:
        h_id = int(
            np.where(np.log10(mass_each_halo) == min(np.log10(mass_each_halo), key=lambda x: abs(x - truths_right[i])))[
                0])
        pynbody.analysis.halo.center(ic[ids_right[i]], vel=False, wrap=True)
        prog_particles = b(h[h_id])
        r_right.append(list(prog_particles['r']))

    np.save("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/seed_20/r_right.npy",
            np.concatenate(r_right))
    np.save("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/seed_20/r_wrong.npy",
            np.concatenate(r_wrong))

    r_right = np.concatenate(r_right)
    r_wrong = np.concatenate(r_wrong)

_ = plt.hist(np.concatenate((r_right, r_wrong)), bins=30, density=True, label="all", histtype="step")
__ = plt.hist(r_right, bins=_[1], density=True, label=r"$\log \left( M_\mathrm{predicted}/M_\mathrm{true} \right) > "
                                                      r"-1$", histtype="step")
__ = plt.hist(r_wrong, bins=_[1], density=True, label=r"$\log \left( M_\mathrm{predicted}/M_\mathrm{true} \right)<= "
                                                      r"-1$", histtype="step")
plt.legend(loc="best")
plt.title("Outskirt particles in high mass halos")


# subhalos
import matplotlib.pyplot as plt

subh = f.halos(subs=True)
sub_grp = f['subh_grp'][np.argsort(f['iord'])]
ind_subhalos = (r/r_vir > 0.8) & (t >= 13) & (sub_grp[ids] != -1)
ind_not_subhalos = (r/r_vir > 0.8) & (t >= 13) & (sub_grp[ids] == -1)
_ = plt.hist((p - t)[(r/r_vir > 0.8) & (t >= 13)], bins=30, histtype="step", density=True, color="k", label="all")
_ = plt.hist((p - t)[ind_subhalos], bins=_[1], histtype="step", density=True, label="in subhalo")
__ = plt.hist((p - t)[ind_not_subhalos], bins=_[1], histtype="step", density=True, label="not in subhalo")
plt.legend(loc="best", fontsize=14)
plt.savefig("/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/seed_20"
            "/outskirt_particles_subhalos.png")

