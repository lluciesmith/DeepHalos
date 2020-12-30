import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
import pynbody
import time
from multiprocessing import Pool
from collections import OrderedDict


if __name__ == "__main__":
    sims = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]

    for i in range(len(sims)):
        sim = sims[i]

        path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
        saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

        f = pynbody.load(path_sim + "output/snapshot_007")
        f.physical_units()

        # load halo masses each halo

        print("Loading the halos...")

        h = f.halos()
        assert h._ordered == False
        masses = np.load(saving_path + "mass_Msol_each_halo_sim_" + sim + ".npy")


        def halo_iord_with_pool(num_halos):
            ids = list(np.arange(num_halos))
            pool = Pool(80)
            masses = pool.map(get_halo_iord, ids)
            pool.close()
            return masses

        def get_halo_iord(halo_id):
            return h[halo_id]["iord"]


        t0 = time.time()
        halos_iord = halo_iord_with_pool(len(h))
        t1 = time.time()
        print("Done halos iord loop, it took " + str((t1 - t0) / 60) + " minutes.")

        m_dict = OrderedDict(zip(np.arange(len(h)), masses))
        iord_dict = OrderedDict(zip(np.arange(len(h)), halos_iord))

        t0 = time.time()
        halo_mass_ids = np.zeros(len(f), )
        for key in iord_dict.keys():
            halo_mass_ids[iord_dict[key]] = m_dict[key]
        t1 = time.time()
        print("Done halos mass particles loop, it took " + str((t1 - t0) / 60) + " minutes.")

        np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids)


