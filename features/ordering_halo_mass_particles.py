import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
from mlhalos import parameters
import pynbody
import time


if __name__ == "__main__":
    sims = ["6", "7", "8", "9", "10"]
    for i in range(5):
        sim = sims[i]

        path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
        saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

        f = pynbody.load(path_sim + "output/snapshot_007")
        f.physical_units()

        # get halo masses each particles
        print("Loading the halos...")
        t0 = time.time()
        h = f.halos()
        t1 = time.time()
        print("Loading halos took " + str((t1 - t0)/60) + " minutes.")
        print("Done loading halos.")
        assert h._ordered == False
        #h_id = f['grp']

        halo_ids = np.arange(len(h))
        halo_mass_ids = np.zeros(len(f),)

        for hi in halo_ids:
            ids_in_halos = h[hi]["iord"]
            halo_mass_ids[ids_in_halos] = h[hi]['mass'].sum()

        np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids)

        # # Save halo mass of particles and select training samples
        # particle_ids = f["iord"]
        #
        # ind = np.argsort(particle_ids)
        # assert np.allclose(particle_ids[ind], np.arange(len(f)).astype("int"))
        #
        # np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids[ind])
