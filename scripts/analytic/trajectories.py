import sys
sys.path.append("/home/lls/mlhalos_code")
import numpy as np
from mlhalos import parameters
from mlhalos import density
import pynbody
import gc

if __name__ == "__main__":
    for sim in ["22", "23", "24"]:
        saving_path5 = "/share/data1/lls/standard_reseed%i/" % sim
        path_simulation5 = "/share/hypatia/lls/simulations/standard_reseed%i/" % sim
        initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_simulation5 + "IC.gadget2",
                                                                load_final=False, min_halo_number=0, max_halo_number=400,
                                                                min_mass_scale=3e10, max_mass_scale=1e15)
        # Get features for each particles
        Dc = density.DensityContrasts(initial_parameters=initial_params, num_filtering_scales=50, path="/home/lls")
        density_cont = Dc.density_contrasts
        np.save(saving_path5 + "density_contrasts.npy", density_cont)
