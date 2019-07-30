import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
sys.path.append("/home/lls/")
# import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
# sys.path.append("/Users/lls/Documents/Projects/")
from mlhalos import parameters
from DeepHalos import subboxes as subb
from multiprocessing import Pool
import os
import gc
import re


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


if __name__ == "__main__":
    path_sim = "/share/hypatia/app/luisa/reseed2/"
    saving_path = "/share/hypatia/lls/deep_halos/reseed_2/training_set/"
    halo_mass = np.load("/share/data1/lls/reseed50_2/halo_mass_particles.npy")

    initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC_doub_z99_256.gadget3",
                                                            final_snapshot=path_sim + "snapshot_099",
                                                            load_final=True)
    sub_in = subb.Subboxes(initial_params, subbox_shape=(51, 51, 51))
    p_ids = np.where(halo_mass > 0)[0]

    subset_ids = np.random.choice(p_ids, 30000, replace=False)
    np.save(saving_path + "../subset_30000_ids.npy", subset_ids)
    saved_ids = []

    pool = Pool(processes=80)
    pool.map(compute_and_save_subbox_particle, subset_ids)
    pool.close()

    np.savetxt("/share/hypatia/lls/deep_halos/reseed_2/saved_ids_training_set.txt",
               np.array(saved_ids), fmt="%i", delimiter=",")

    # val_ids = []
    # regex = re.compile(r'\d+')
    # for filename in os.listdir(saving_path):
    #     val_ids.append(int(regex.findall(filename)[-1]))
    #
    # val_ids = np.array(val_ids)
    # print(len(val_ids))
    # np.save("/share/hypatia/lls/deep_halos/reseed_2/reseed2_subboxes_ids.npy", val_ids)
    #
    #
    # if __name__ == "__main__":
    #     path_sim = "/share/hypatia/app/luisa/reseed2/"
    #     saving_path = "/share/data2/lls/deep_halos/reseed_2/subboxes"
    #
    #     initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC_doub_z99_256.gadget3",
    #                                                             final_snapshot=path_sim + "snapshot_099",
    #                                                             load_final=True)
    #     sub_in = subb.Subboxes(initial_params, subbox_shape=(51, 51, 51))
    #
    #     halo_mass = np.load("/share/data1/lls/reseed50_2/halo_mass_particles.npy")
    #     p_ids = np.where(halo_mass > 0)[0]
    #     subset_ids = np.random.choice(p_ids, 100000, replace=False)
    #     np.save(saving_path + "subset_50000_ids.npy", subset_ids)
    #     sub_in.compute_and_save_subboxes(subset_ids, saving_path)

