import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
sys.path.append("/home/lls/")
# import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
# sys.path.append("/Users/lls/Documents/Projects/")
from mlhalos import parameters
from DeepHalos import subboxes as subb
from multiprocessing import Pool
import gc


def compute_and_save_subbox_particle(particle_id):
    delta_sub = sub_in.get_qty_in_subbox(particle_id)
    np.save(saving_path + str(particle_id) + "/subbox_51_particle_" + str(particle_id) + ".npy", delta_sub)
    del delta_sub
    gc.collect()

if __name__ == "__main__":
    path_sim = "/share/hypatia/app/luisa/standard_reseed4/"
    saving_path = "/share/hypatia/lls/deep_halos/reseed_4/subboxes"
    halo_mass = np.load("/share/data1/lls/standard_reseed4/halo_mass_particles.npy")

    initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC.gadget3",
                                                            final_snapshot=path_sim + "snapshot_099",
                                                            load_final=True)
    sub_in = subb.Subboxes(initial_params, subbox_shape=(51, 51, 51))
    p_ids = np.where(halo_mass > 0)[0]

    subset_ids = np.random.choice(p_ids, 100000, replace=False)
    np.save(saving_path + "subset_100000_ids.npy", subset_ids)

    pool = Pool(processes=60)
    pool.map(compute_and_save_subbox_particle, subset_ids)
    pool.close()