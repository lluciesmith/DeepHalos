import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
sys.path.append("/home/lls/")
# import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
# sys.path.append("/Users/lls/Documents/Projects/")
from mlhalos import parameters
from DeepHalos import subboxes as subb
from multiprocessing import Pool
import gc
import os


def compute_and_save_subbox_particle(particle_id):
    try:
        delta_sub = sub_in.get_qty_in_subbox(particle_id)
        saved_ids.append(particle_id)
        os.makedirs(saving_path + str(particle_id))
        np.save(saving_path + str(particle_id) + "/subbox_51_particle_" + str(particle_id) + ".npy", delta_sub)
        del delta_sub
        gc.collect()

    except ValueError:
        print("This failed for particle " + str(particle_id))

############## SIMS 3,4,5 ##############


path_sims = ["/share/hypatia/app/luisa/standard_reseed3/", "/share/hypatia/app/luisa/standard_reseed4/",
             "/share/hypatia/app/luisa/standard_reseed5/"]
paths_ids = ["/share/hypatia/lls/deep_halos/reseed_3/", "/share/hypatia/lls/deep_halos/reseed_4/",
             "/share/hypatia/lls/deep_halos/reseed_5/"]
filenames = ["reseed_3_random_training_set.txt", "reseed_4_random_training_set.txt", "reseed_5_random_training_set.txt"]

for i in range(3):
    ps = path_sims[i]
    pi = paths_ids[i]
    f = filenames[i]

    saving_path = pi + "z2_subboxes/"
    initial_params = parameters.InitialConditionsParameters(initial_snapshot=ps + "snapshot_049",
                                                            load_final=False)
    sub_in = subb.Subboxes(initial_params, subbox_shape=(51, 51, 51))

    ids = np.loadtxt(pi + f, dtype='int', delimiter=",")
    saved_ids = []

    pool = Pool(processes=80)
    pool.map(compute_and_save_subbox_particle, ids)
    pool.close()

    np.save(pi + "test_" + f, saved_ids)

    del saved_ids, initial_params, sub_in, ps, saving_path, pi, f


####### TRAINING SIM #########

path_sim = "/home/lls/stored_files/Nina-Simulations/double/"
path_ids = "/share/hypatia/lls/deep_halos/training_sim/"

saving_path = path_ids + "z2_subboxes/"
initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "snapshot_054", load_final=False)
sub_in = subb.Subboxes(initial_params, subbox_shape=(51, 51, 51))

ids = np.loadtxt(path_ids + "training_sim_random_training_set.txt", dtype='int', delimiter=",")
saved_ids = []

pool = Pool(processes=80)
pool.map(compute_and_save_subbox_particle, ids)
pool.close()

np.save(path_ids + "test_training_sim_random_training_set.txt", saved_ids)

del saved_ids, initial_params, sub_in, path_sim, saving_path, path_ids

