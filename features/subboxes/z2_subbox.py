import pynbody
import numpy as np
from multiprocessing import Pool
import gc
import os


def get_sph_particle(sim, particle_id, width=200, resolution=51, wrap=True):
    pynbody.analysis.halo.center(sim[particle_id], vel=False, wrap=wrap)
    subbox_sph = get_sph_on_3dgrid(sim, width=width, resolution=resolution)
    return subbox_sph


def get_sph_on_3dgrid(sim, width=200, resolution=51):
    x2 = width/2
    xy_units = sim["pos"].units
    grid_data = pynbody.sph.to_3d_grid(sim, qty="delta", nx=resolution, x2=x2, xy_units=xy_units)
    return grid_data


def compute_and_save_subbox_particle(particle_id):
    try:
        delta_sub = get_sph_particle(sim, particle_id, width=200, resolution=51)
        saved_ids.append(particle_id)
        os.makedirs(saving_path + str(particle_id))
        np.save(saving_path + str(particle_id) + "/subbox_51_particle_" + str(particle_id) + ".npy", delta_sub)
        del delta_sub
        gc.collect()

    except ValueError:
        print("This failed for particle " + str(particle_id))


def delta_property(snapshot):
    rho = snapshot["rho"]
    mean_rho = pynbody.analysis.cosmology.rho_M(snapshot, unit=rho.units)
    snapshot["delta"] = rho / mean_rho
    snapshot["delta"].units = "1"
    return snapshot["delta"]

############## SIMS 3,4,5 ##############


# path_sims = ["reseed/", "reseed2/", "standard_reseed3/", "standard_reseed4/", "standard_reseed5/"]
# paths_ids = ["reseed_1/", "reseed_2/", "reseed_3/", "reseed_4/", "reseed_5/"]
# filenames = ["reseed_1","reseed_2", "reseed_3", "reseed_4", "reseed_5"]
#
# for i in range(len(path_sims)):
#     ps = "/share/hypatia/app/luisa/" + path_sims[i]
#     pi = "/share/hypatia/lls/deep_halos/" + paths_ids[i]
#     f = filenames[i] + "_random_training_set.txt"
#
#     saving_path = pi + "z2_subboxes/"
#
#     sim = pynbody.load(ps + "snapshot_049")
#     sim.physical_units()
#     d = delta_property(sim)
#
#     ids = np.loadtxt(pi + f, dtype='int', delimiter=",")
#     saved_ids = []
#
#     pool = Pool(processes=5)
#     pool.map(compute_and_save_subbox_particle, ids)
#     pool.close()
#
#     np.save(pi + "test_" + f, saved_ids)
#
#     del ps, pi, f, saving_path, sim, d, ids, saved_ids


####### TRAINING SIM #########

path_sim = "/home/lls/stored_files/Nina-Simulations/double/"
path_ids = "/share/hypatia/lls/deep_halos/training_sim/"

saving_path = path_ids + "z2_subboxes/"

sim = pynbody.load(path_sim + "snapshot_054")
sim.physical_units()
d = delta_property(sim)

ids = np.loadtxt(path_ids + "training_sim_random_training_set.txt", dtype='int', delimiter=",")
saved_ids = []

for pid in ids:
    compute_and_save_subbox_particle(pid)

# pool = Pool(processes=1)
# pool.map(compute_and_save_subbox_particle, ids)
# pool.close()

np.save(path_ids + "test_training_sim_random_training_set.txt", saved_ids)

