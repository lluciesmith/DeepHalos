import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
import pynbody
import time
from multiprocessing import Pool


if __name__ == "__main__":
    # # sims = ["6", "7", "8", "9", "10"]
    # sims = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    # for i in range(len(sims)):
    #     sim = sims[i]

    sim = "11"
    
    path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
    saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

    f = pynbody.load(path_sim + "output/snapshot_007")
    f.physical_units()

    # Halo mass

    def get_halo_mass(halo_id):
        halo = h[halo_id]
        return float(halo['mass'].sum())


    def get_mass_with_pool(num_halos):
        ids = list(np.arange(num_halos))
        pool = Pool(40)
        masses = pool.map(get_halo_mass, ids)
        pool.close()
        return masses

    def get_mass_each_halo(halo_catalogue):
        t0 = time.time()
        masses = get_mass_with_pool(len(halo_catalogue))
        t1 = time.time()
        print("Loading halo masses took " + str((t1 - t0) / 60) + " minutes.")
        np.save(saving_path + "mass_Msol_each_halo_sim_" + sim + ".npy", masses)

    # def get_halo_mass_each_particle(halo_masses, snapshot, halo_catalogue):
    #     halo_mass_ids = np.zeros(len(snapshot), )
    #     for i, hid in enumerate(halo_catalogue):
    #         halo_mass_ids[hid["iord"]] = halo_masses[i]
    #
    #     np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids)
    #     del halo_catalogue

    # get halo masses each halo

    print("Loading the halos...")

    h = f.halos()
    assert h._ordered == False

    t0 = time.time()
    masses = get_mass_with_pool(len(h))
    t1 = time.time()
    print("Loading halo masses took " + str((t1 - t0)/60) + " minutes.")
    np.save(saving_path + "mass_Msol_each_halo_sim_" + sim + ".npy", masses)

    # get halo masses each particles

    halo_mass_ids = np.zeros(len(f),)
    for i, hid in enumerate(h):
        halo_mass_ids[hid["iord"]] = masses[i]

    np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids)

    del h
