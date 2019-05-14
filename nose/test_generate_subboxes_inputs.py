"""

Here, I am comparing current method to generate input subboxes with
old one where I was defining subboxes based on the positions of
particles.
This was sometimes annoying because particles are slights displaced
and not exactly on a grid. But it is useful as a cross check!

"""


import numpy as np
import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
from mlhalos import parameters
import scipy.special
import gc
from .. import input_subboxes


def delta_property(snapshot):
    rho = snapshot["rho"]
    mean_rho = ic.get_mean_matter_density_in_the_box(snapshot, units=str(snapshot["pos"].units))
    snapshot["delta"] = rho / mean_rho
    return snapshot["delta"]


def get_distances_from_particle_id(snapshot, particle_id, in_3d=False):
    x_p, y_p, z_p = snapshot["x"][particle_id], snapshot["y"][particle_id], snapshot["z"][particle_id]
    x_all, y_all, z_all = snapshot["x"] - x_p, snapshot["y"] - y_p, snapshot["z"] - z_p
    if in_3d is True:
        return x_all.reshape(256, 256, 256), y_all.reshape(256, 256, 256), z_all.reshape(256, 256, 256)
    else:
        return x_all, y_all, z_all


def get_particle_ids_in_subbox_from_distances_0(snapshot, x_distance, y_distance, z_distance, shape=(9 , 9 ,9),
                                                b=0.005):
    shape_box = int(scipy.special.cbrt(len(snapshot)))
    gridsize = snapshot.properties["boxsize"] / shape_box
    len_subbox = shape[0] * gridsize

    n = float(gridsize)*b
    if len(x_distance.shape) == 3:
        idx = np.where((x_distance <= len_subbox/2 - n) & (x_distance >= -len_subbox/2 - n ) &
                       (y_distance <= len_subbox/2 - n) & (y_distance >= -len_subbox/2 - n ) &
                       (z_distance <= len_subbox/2 - n) & (z_distance >= -len_subbox/2 - n ))
    else:
        idx = np.where((x_distance <= len_subbox/2 - n) & (x_distance >= -len_subbox/2 - n ) &
                       (y_distance <= len_subbox/2 - n) & (y_distance >= -len_subbox/2 - n ) &
                       (z_distance <= len_subbox/2 - n) & (z_distance >= -len_subbox/2 - n ))[0]
    return idx


def get_particle_ids_in_subbox_from_distances(snapshot, x_distance, y_distance, z_distance, shape=(9 , 9 ,9), b=0.005):

    idx = get_particle_ids_in_subbox_from_distances_0(snapshot, x_distance, y_distance, z_distance, shape=shape, b=b)
    try:
        assert len(idx) == shape[0]**3

    except AssertionError:
        print("Since ids found are %i instead of %i try different distances from sub-box edges" % (len(idx),
                                                                                                   shape[0]**3))
        if len(idx) > shape[0]**3:
            b1 = [b*1.5, b*1.6, b*1.8, b*1.9, b*2]
            for threshold in b1:
                idx = get_particle_ids_in_subbox_from_distances_0(snapshot, x_distance, y_distance, z_distance,
                                                                  shape=shape, b=threshold)
                if len(idx) == shape[0]**3:
                    break

        elif len(idx) < shape[0]**3:
            b1 = [b/1.4, b/2, b/2.5, b/3, 0, -0.005, -0.1]
            for threshold in b1:
                idx = get_particle_ids_in_subbox_from_distances_0(snapshot, x_distance, y_distance, z_distance,
                                                                  shape=shape, b=threshold)
                if len(idx) == shape[0]**3:
                    break

        assert len(idx) == shape[0] ** 3, "Ids found are %i instead of %i" % (len(idx), shape[0] ** 3)
    return idx


def get_particle_ids_in_subbox_around_particleid(snapshot, particle_id, shape=(9,9,9), in_3d=False):
    x_all, y_all, z_all = get_distances_from_particle_id(snapshot, particle_id, in_3d=in_3d)
    idx = get_particle_ids_in_subbox_from_distances(snapshot, x_all, y_all, z_all, shape=shape)

    mid_point = int((shape[0] - 1)/2)
    assert idx.reshape(shape)[mid_point, mid_point, mid_point] == particle_id, \
        "Particle ID is not at center of the box!"
    return idx


def densities_in_subbbox_around_particleid(snapshot, particle_id, qty="delta", shape=(9, 9, 9), in_3d=False):
    idx = get_particle_ids_in_subbox_around_particleid(snapshot, particle_id, shape=shape, in_3d=in_3d)
    rho = snapshot[qty]
    return rho[idx].reshape(shape)


def densities_subbbox_particles_excluding_edges(snapshot, particle_id, qty="delta", shape=(9, 9, 9), in_3d=False):
    shape_box = int(scipy.special.cbrt(len(snapshot)))
    pos_particle = snapshot["pos"][particle_id]

    pos_max = snapshot["pos"].max()
    pos_min = snapshot["pos"].min()

    n = (pos_max - pos_min)/shape_box
    lim = shape[0] + 1
    if any(pos_particle >= pos_max - lim*n) or any(pos_particle <= pos_min + lim*n):
        return np.zeros(shape)
    else:
        rho = densities_in_subbbox_around_particleid(snapshot, particle_id, qty=qty, shape=shape, in_3d=in_3d)
        return rho


def subboxes_around_particles(snapshot, particles, qty="delta", shape=(9, 9, 9), in_3d=False):
    inputs = np.zeros((len(particles), shape[0], shape[1], shape[2]))

    for i in range(len(particles)):
        print("Particle " + str(i) + ": "+ str(particles[i]))
        if i == len(particles)/2:
            print("Half way")
        d = densities_subbbox_particles_excluding_edges(snapshot, particles[i], qty=qty, shape=shape, in_3d=in_3d)
        inputs[i] = d
        del d
        gc.collect()

    return inputs


def test_generate_subboxes_around_particles():
    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/mlhalos_files/")
    d = delta_property(ic.initial_conditions)

    halo_mass = np.load("/Users/lls/Documents/mlhalos_files/halo_mass_particles.npy")
    ids_in_halo = np.where(halo_mass > 0)[0]

    n = np.random.choice(ids_in_halo, 1000, replace=False)
    n_delta_old = subboxes_around_particles(ic.initial_conditions, n, shape=(17, 17, 17))
    a = np.unique(np.where(n_delta_old != 0)[0])

    n_delta_new = input_subboxes.delta_in_subboxes_around_particles(ic.initial_conditions, n[a],
                                                                    qty="delta", subbox_shape=(9, 9, 9))
    np.testing.assert_allclose(n_delta_old[a], n_delta_new)

