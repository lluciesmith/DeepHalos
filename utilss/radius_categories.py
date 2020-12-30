import numpy as np

import numpy as np
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
sys.path.append('/Users/lls/Documents/mlhalos_code/scripts')
from radius_analysis import radii_fraction_in as rfi
from mlhalos import parameters
from utils import radius_func as rad
from collections import OrderedDict
import pynbody


def find_indices_of_particle_ids_in_halo(all_halos_particles, halo_ID):
    pos = np.in1d(all_halos_particles, halo_ID)
    return pos


def get_radius_particles_of_halo(all_radii_particles, all_halos_particles, halo_ID, pos=None):
    if pos is None:
        pos = find_indices_of_particle_ids_in_halo(all_halos_particles, halo_ID)
    r = all_radii_particles[pos]
    return r


def fraction_virial_radius(all_particles, all_radii_particles, all_halos_particles, virial_radii):
    radius_fraction = np.zeros((len(all_particles),))
    set_halos = np.unique(all_halos_particles)

    for i in range(len(set_halos)):
        halo_ID = set_halos[i]
        virial_radius = virial_radii[i]
        print(halo_ID)
        pos = find_indices_of_particle_ids_in_halo(all_halos_particles, halo_ID)
        if halo_ID == 336:
            radius_fraction[pos] == np.NaN
        else:
            rad_particles = get_radius_particles_of_halo(all_radii_particles, all_halos_particles, halo_ID, pos=pos)
            radius_fraction[pos] = rad_particles/virial_radius
    return radius_fraction


if __name__ == "__main__":
    f = pynbody.load("/Users/lls/Documents/mlhalos_files/reseed7/snapshot_007")
    f.physical_units()
    h = f.halos()

    ids = f["iord"]

    dict_i = OrderedDict()
    dict_i.items = ["%s" % particle_id for particle_id in ids]

    # which halo ID has mass M=10^11 Msol in this sim? HALO ID:

    # Compute virial radius of every halos

    ids_0 = np.load("ids_0.npy")
    ind = np.where(f[ids_0]["grp"] >= 2436)[0]
    hid = f[ids_0[ind]]["grp"]

    virial_h = np.zeros(len(hid))

    for i in np.unique(hid):
        try:
            ind_i = np.where(hid == i)[0]
            pynbody.analysis.halo.center(h[i], vel=False, wrap=True)
            virial_h[ind_i] = pynbody.analysis.halo.virial_radius(h[i], overden=200)
        except:
            pass


    r_prop = np.load("/Users/lls/Documents/mlhalos_files/stored_files/all_out/radii_files/"
                     "radii_properties_all_ids_in_halos.npy")
    ids_in = r_prop[:,0].astype("int")

    testing_ids = np.load("/Users/lls/Desktop/testing_ids_valid_mass_range.npy")
    testing_ids_in_r = np.in1d(ids_in, testing_ids)

    r_testing = r_prop[testing_ids_in_r]
    ids_testing = r_testing[:,0].astype("int")
    radius_testing = r_testing[:,1]
    halos_testing = ic.final_snapshot[ids_testing]["grp"]

    ost_400 = np.where(halos_testing >= 400)[0]
    vir_r = []
    for halo_ID in set(halos_testing[ost_400]):
        print(halo_ID)
        vir_r.append(float(rad.virial_radius(halo_ID, f=f, h=h)))

    np.save("/Users/lls/Desktop/virial_radii_400_to_2436.npy", np.array(vir_r))

    radius_fraction_in = fraction_virial_radius(ids_testing[ost_400], radius_testing[ost_400], halos_testing[ost_400],
                                                vir_r)

    r_testing[ost_400, 2] = radius_fraction_in
    np.save("/Users/lls/Documents/mlhalos_files/stored_files/all_out/radii_files"
            "/correct_radii_prop_testing_ids_valid_mass_range.npy",
            r_testing)


    ####################
    import pynbody

    f = pynbody.load("/Users/lls/Documents/mlhalos_files/Nina-Simulations/double/snapshot_104")
    f.physical_units()
    h = f.halos(make_grp=True)
    ids_0 = np.load("ids_0.npy")
    ind = np.where(f[ids_0]["grp"] >= 2436)[0]
    hid = f[ids_0[ind]]["grp"]

    virial_h = np.zeros(len(hid))

    for i in np.unique(hid):
        try:
            ind_i = np.where(hid == i)[0]
            pynbody.analysis.halo.center(h[i], vel=False, wrap=True)
            virial_h[ind_i] = pynbody.analysis.halo.virial_radius(h[i], overden=200)
        except:
            pass

    np.save("/Users/lls/Desktop/virial_h_ind.npy", virial_h)

    r_prop = np.load(
        "/Users/lls/Documents/mlhalos_files/stored_files/all_out/radii_files/correct_radii_properties_all_ids_in_halos_upto_2436.npy")
    ids_rprop = r_prop[:, 0].astype("int")
    for i in range(len(ids_0[ind])):
        id_i = ids_0[ind][i]
        n = ids_rprop == id_i
        r_prop[n, 2] = r_prop[n, 1]/virial_h[i]

    np.save("/Users/lls/Documents/mlhalos_files/stored_files/all_out/radii_files/"
            "correct_radii_properties_ids_0.npy", r_prop)

