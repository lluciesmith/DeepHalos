import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
import pynbody
import time
from multiprocessing import Pool
from collections import OrderedDict


if __name__ == "__main__":
    sims = ["L200_N1024_genetIC", "L200_N1024_genetIC2", "L200_N1024_genetIC3"]

    for sim in sims:
        saving_path = "/share/hypatia/lls/deep_halos/" + sim

        path = "/mnt/beegfs/work/ati/pearl037/" + sim + "/snapshots/"
        fp = pynbody.load(path + "snapshot_000.hdf5")
        orig_unit = pynbody.units.g / pynbody.units.h * float(fp.infer_original_units('kg'))
        units_ratio = pynbody.units.Msol.ratio(orig_unit, h=fp.properties['h'])
        ids = fp['iord']

        h = h5py.File(path + "fof_subhalo_tab_000.hdf5", "r")
        offset = h['Group']['GroupOffsetType'][:,1]
        length = h['Group']['GroupLen'][:]

        m = h['Group']['GroupMass'][:]
        halo_mass_ids = np.zeros((len(ids)))
        for i in np.arange(len(length)):
            ind = ids[offset[i]: offset[i] + length[i]]
            halo_mass_ids[ind] = m[i] / units_ratio

        np.save(saving_path + "reseed" + sim + "_halo_mass_particles.npy", halo_mass_ids)
        del fp, h, halo_mass_ids