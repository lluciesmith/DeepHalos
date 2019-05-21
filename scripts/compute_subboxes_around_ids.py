import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
sys.path.append("/home/lls/")
# import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
# sys.path.append("/Users/lls/Documents/Projects/")
from mlhalos import parameters
from DeepHalos import subboxes as subb


if __name__ == "__main__":
    path = "/home/lls/stored_files"
    # path = "/Users/lls/Documents/mlhalos_files/"
    saving_path = "/share/data2/lls/deep_halos/subboxes"

    ic = parameters.InitialConditionsParameters(path=path)
    sub_in = subb.Subboxes(ic, subbox_shape=(51, 51, 51))

    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
    p_ids = np.where(halo_mass > 0)[0]
    sub_in.compute_and_save_subboxes(p_ids, saving_path)

