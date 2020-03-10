import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
import pynbody
import time
from multiprocessing import Pool
#
#
# def get_halo_mass(halo_id):
#     halo = h[halo_id]
#     return float(halo['mass'].sum())
#
#
# def get_mass_with_pool(num_halos=10):
#     ids = list(np.arange(num_halos))
#     pool = Pool(40)
#     masses = pool.map(get_halo_mass, ids)
#     pool.close()
#     return masses
#
# f = pynbody.load("/share/hypatia/lls/simulations/standard_reseed6/output/snapshot_007")
# f.physical_units()
#
# t0 = time.time()
# h = f.halos()
# m1 = get_mass_with_pool(len(h))
# t1 = time.time()
# print("Parallelised loading masses took " + str((t1 - t0)/60) + " minutes.")


if __name__ == "__main__":
    sims = ["6", "7", "8", "9", "10"]
    for i in range(len(sims)):
        sim = sims[i]

        path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
        saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

        f = pynbody.load(path_sim + "output/snapshot_007")
        f.physical_units()


        def get_halo_mass(halo_id):
            halo = h[halo_id]
            return float(halo['mass'].sum())


        def get_mass_with_pool(num_halos=10):
            ids = list(np.arange(num_halos))
            pool = Pool(40)
            masses = pool.map(get_halo_mass, ids)
            pool.close()
            return masses

        # get halo masses each halo

        print("Loading the halos...")
        t0 = time.time()

        h = f.halos()
        assert h._ordered == False

        masses = get_mass_with_pool(len(h))

        t1 = time.time()
        print("Loading halos took " + str((t1 - t0)/60) + " minutes.")


        # t0 = time.time()
        # masses = np.array([hi['mass'].sum() for hi in h])
        # t1 = time.time()
        # print("Loading halo masses took " + str((t1 - t0)/60) + " minutes.")
        np.save(saving_path + "mass_Msol_each_halo_sim_" + sim + ".npy", masses)

        # get halo masses each particles

        halo_mass_ids = np.zeros(len(f),)
        for i, hid in enumerate(h):
            halo_mass_ids[hid["iord"]] = masses[i]

        np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids)

        del h

