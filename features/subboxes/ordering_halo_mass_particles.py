import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
from mlhalos import parameters


if __name__ == "__main__":
    sims = ["6", "7", "8", "9", "10"]
    for i in range(5):
        sim = sims[i]

        path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
        saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

        initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC.gadget2",
                                                                final_snapshot=path_sim + "output/snapshot_007",
                                                                load_final=True)

        # Save halo mass of particles and select training samples
        particle_ids = initial_params.final_snapshot["iord"]
        halo_mass_ids = np.load(saving_path + "reseed" + sim + "_halo_mass_particles.npy")

        ind = np.argsort(particle_ids)
        assert np.allclose(particle_ids[ind], np.arange(256**3).astype("int"))

        np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids[ind])
