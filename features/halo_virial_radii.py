import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
import pynbody
import time
from numba import njit, prange
from multiprocessing import Pool


if __name__ == "__main__":
    #sims = ["6", "7", "8", "9", "10"]
    sims = ["6"]
    for i in range(len(sims)):
        sim = sims[i]

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

        def get_halo_mass_each_particle(halo_masses, snapshot, halo_catalogue):
            halo_mass_ids = np.zeros(len(snapshot), )
            for i, hid in enumerate(halo_catalogue):
                halo_mass_ids[hid["iord"]] = halo_masses[i]

            np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids)
            del halo_catalogue

        # Virial radius


        def get_virial_radius_halo(halo_id):
            if halo_id == 200:
                print("Halo ID 200")
            try:
                pynbody.analysis.halo.center(h[halo_id], vel=False, wrap=True)
                rvir = pynbody.analysis.halo.virial_radius(h[halo_id], overden=200)
            except:
                print("Halo " + str(halo_id) + " didn't work.")
                rvir = 0
            return float(rvir)

        def get_virial_r_with_pool(num_halos=10):
            ids = list(np.arange(num_halos))
            pool = Pool(40)
            masses = pool.map(get_virial_radius_halo, ids)
            pool.close()
            return masses

        def get_halo_virial_radius(halo_catalogue):
            t0 = time.time()
            radii = get_virial_r_with_pool(len(halo_catalogue))
            t1 = time.time()
            print("Loading halo masses took " + str((t1 - t0) / 60) + " minutes.")
            np.save(saving_path + "virial_radius_each_halo_sim_" + sim + ".npy", radii)
            return radii


        def get_radius_in_halo_each_particle(virial_radii, snapshot, halo_catalogue):
            # virial_radii_ids = np.zeros(len(snapshot), )
            radii_ids = np.zeros(len(snapshot), )

            all_halos = np.arange(len(virial_radii))
            halos_ok = all_halos[np.in1d(all_halos, np.where(virial_radii != 0)[0])]

            for i, HID in halos_ok:
                if HID % 1000 == 0:
                    print("Halo ID " + str(HID))
                pynbody.analysis.halo.center(h[HID], vel=False, wrap=True)
                ind = np.asarray(halo_catalogue[HID]["iord"])

                # virial_radii_ids[ind] = virial_radii[i]
                radii_ids[ind] = halo_catalogue[HID]['r']

            # np.save(saving_path + "reseed" + sim + "_virial_radius_particles.npy", virial_radii_ids)
            np.save(saving_path + "reseed" + sim + "radius_in_halo_particles.npy", radii_ids)
            return radii_ids


        # get virial radii for each halo and then for each particle

        print("Loading the halos...")

        h = f.halos()
        assert h._ordered == False

        # r_halos = get_halo_virial_radius(h)

        r_halos = np.load(saving_path + "virial_radius_each_halo_sim_" + sim + ".npy")
        r_particles = get_radius_in_halo_each_particle(r_halos, f, h)

