import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
from mlhalos import parameters
import pynbody


if __name__ == "__main__":
    sims = ["6", "7", "8", "9", "10"]
    for i in range(5):
        sim = sims[i]

        path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
        saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

        f = pynbody.load(path_sim + "output/snapshot_007")
        f.physical_units()

        # get halo masses each particles

        h = f.halos(order=False, make_grp=True)
        h_id = f['grp']

        halo_ids = np.arange(len(h))
        halo_mass_ids = np.zeros(len(f),)

        for i, hi in enumerate(halo_ids):
            ind = np.where(h_id == hi)[0]
            halo_mass_ids[ind] = h[hi]['mass'].sum()

        # Save halo mass of particles and select training samples
        particle_ids = f["iord"]

        ind = np.argsort(particle_ids)
        assert np.allclose(particle_ids[ind], np.arange(len(f)).astype("int"))

        np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids[ind])
