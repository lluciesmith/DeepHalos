import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
sys.path.append("/home/lls/")
# import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
# sys.path.append("/Users/lls/Documents/Projects/")
from mlhalos import parameters
from DeepHalos import subboxes as subb


if __name__ == "__main__":
    path_sim = "/share/hypatia/app/luisa/reseed2/"
    saving_path = "/share/data2/lls/deep_halos/reseed_2/subboxes"

    initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC_doub_z99_256.gadget3",
                                                            final_snapshot=path_sim + "snapshot_099",
                                                            load_final=True)
    sub_in = subb.Subboxes(initial_params, subbox_shape=(51, 51, 51))

    halo_mass = np.load("/share/data1/lls/reseed50_2/halo_mass_particles.npy")
    p_ids = np.where(halo_mass > 0)[0]
    subset_ids = np.random.choice(p_ids, 50000, replace=False)
    np.save(saving_path + "subset_50000_ids.npy", subset_ids)
    sub_in.compute_and_save_subboxes(subset_ids, saving_path)

