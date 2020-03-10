import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
import pynbody
import time


if __name__ == "__main__":
    sims = ["6", "7", "8", "9", "10"]
    for i in range(len(sims)):
        sim = sims[i]

        path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
        saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

        f = pynbody.load(path_sim + "output/snapshot_007")
        f.physical_units()

        # get halo masses each halo

        print("Loading the halos...")
        t0 = time.time()
        h = f.halos()
        t1 = time.time()
        print("Loading halos took " + str((t1 - t0)/60) + " minutes.")
        print("Done loading halos.")
        assert h._ordered == False

        t0 = time.time()
        masses = np.array([hi['mass'].sum() for hi in h])
        t1 = time.time()
        print("Loading halo masses took " + str((t1 - t0)/60) + " minutes.")
        np.save(saving_path + "mass_Msol_each_halo_sim_" + sim + ".npy", masses)

        # get halo masses each particles

        halo_mass_ids = np.zeros(len(f),)
        for i, hid in enumerate(h):
            halo_mass_ids[hid["iord"]] = masses[i]

        np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids)

