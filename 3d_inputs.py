import numpy as np
import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
from mlhalos import parameters
import scipy.special

### Input

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


def get_particle_ids_in_subbox_from_distances(snapshot, x_distance, y_distance, z_distance, shape=(9 , 9 ,9)):
    shape_box = int(scipy.special.cbrt(len(snapshot)))
    gridsize = snapshot.properties["boxsize"] / shape_box
    len_subbox = shape[0] * gridsize

    n = float(gridsize)*0.005
    if len(x_distance.shape) == 3:
        idx = np.where((x_distance <= len_subbox/2 - n) & (x_distance >= -len_subbox/2 - n ) &
                       (y_distance <= len_subbox/2 - n) & (y_distance >= -len_subbox/2 - n ) &
                       (z_distance <= len_subbox/2 - n) & (z_distance >= -len_subbox/2 - n ))
    else:
        idx = np.where((x_distance <= len_subbox/2 - n) & (x_distance >= -len_subbox/2 - n ) &
                       (y_distance <= len_subbox/2 - n) & (y_distance >= -len_subbox/2 - n ) &
                       (z_distance <= len_subbox/2 - n) & (z_distance >= -len_subbox/2 - n ))[0]
        assert len(idx) == shape[0]**3, "Ids found are %i instead of %i" % (len(idx), shape[0]**3)
    return idx


def get_particle_ids_in_subbox_around_particleid(snapshot, particle_id, shape=(9,9,9), in_3d=False):
    x_all, y_all, z_all = get_distances_from_particle_id(snapshot, particle_id, in_3d=in_3d)
    idx = get_particle_ids_in_subbox_from_distances(snapshot, x_all, y_all, z_all, shape=shape)
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
        if i == len(particles)/2:
            print("Half way")
        inputs[i] = densities_subbbox_particles_excluding_edges(snapshot, particles[i], qty=qty, shape=shape,
                                                                in_3d=in_3d)
    return inputs


def get_output_log_mass(particles, halo_mass=None):
    if halo_mass is None:
        halo_mass = np.load("/Users/lls/Documents/mlhalos_files/halo_mass_particles.npy")
    return np.log10(halo_mass[particles])


if __name__ == "__main__":

    ic = parameters.InitialConditionsParameters(path="/Users/lls/Documents/mlhalos_files/")
    d = delta_property(ic.initial_conditions)

    # halo_mass = np.load("/Users/lls/Documents/mlhalos_files/halo_mass_particles.npy")
    halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
    ids_in_halo = np.where(halo_mass > 0)[0]

    n = np.random.choice(ids_in_halo, 1000)
    np.save("/Users/lls/Documents/deep_halos_files/particle.npy", n)
    print("Computing subboxes for particles")
    n_delta = subboxes_around_particles(ic.initial_conditions, n, shape=(17, 17, 17))
    np.save("/Users/lls/Documents/deep_halos_files/inputs_particles.npy", n)















