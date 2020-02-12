import numpy as np
import sys;

sys.path.append("/home/lls/mlhalos_code/")
sys.path.append("/home/lls/")
# import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
# sys.path.append("/Users/lls/Documents/Projects/")
from mlhalos import parameters
from DeepHalos import subboxes as subb
from multiprocessing import Pool
import gc
import os


def compute_and_save_subbox_particle(id_path_tuple):
    particle_id, saving_path = id_path_tuple

    try:
        delta_sub = sub_in.get_qty_in_subbox(particle_id)
    except:
        print("This failed for particle " + str(particle_id) + "so doing proper SPH")
        delta_sub = sub_in.get_sph_particle(particle_id)

    os.makedirs(saving_path + str(particle_id))
    np.save(saving_path + str(particle_id) + "/subbox_75_particle_" + str(particle_id) + ".npy", delta_sub)
    del delta_sub
    gc.collect()


if __name__ == "__main__":
    path_sim = "/share/hypatia/lls/simulations/standard_reseed6/"

    initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC.gadget2",
                                                            final_snapshot=path_sim + "output/snapshot_007",
                                                            load_final=True)

    # saving_path = "/share/hypatia/lls/deep_halos/reseed_5/training_set/"
    # sub_in = subb.Subboxes(initial_params, subbox_shape=(51, 51, 51))
    # p_ids = np.where(halo_mass > 0)[0]
    #
    # subset_ids = np.random.choice(p_ids, 30000, replace=False)
    # np.save(saving_path + "../subset_30000_ids.npy", subset_ids)
    # saved_ids = []
    #
    # pool = Pool(processes=60)
    # pool.map(compute_and_save_subbox_particle, subset_ids)
    # pool.close()
    #
    # np.savetxt("/share/hypatia/lls/deep_halos/reseed_5/saved_ids_training_set.txt",
    #            np.array(saved_ids), fmt="%i", delimiter=",")

    # Sub-boxes of shape (75, 75, 75)

    saving_path = "/share/hypatia/lls/deep_halos/reseed_6/training_set_res75/"
    sub_in = subb.Subboxes(initial_params, subbox_shape=(75, 75, 75))
    subset_ids = np.loadtxt("/share/hypatia/lls/deep_halos/reseed_6/reseed_6_random_training_set.txt", dtype="int",
                            delimiter=",")

    pool = Pool(processes=80)
    pool.map(compute_and_save_subbox_particle, (subset_ids, saving_path))
    pool.close()
