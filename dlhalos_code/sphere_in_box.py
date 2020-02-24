import numpy as np
import gc


def get_lagrangian_radius(halo_mass_particle):
    rho_mean = 38086624.18620001  # this is the mean matter density of the Universe in Msol kpc**-3

    r_particle = ((halo_mass_particle / rho_mean) / (4 / 3 * np.pi)) ** (1 / 3)
    return r_particle


def get_sphere_in_box(halo_mass_particle):
    # Halo masses should be in Msol
    r_halo = get_lagrangian_radius(halo_mass_particle)  # in kpc

    l_pixel = 50 * 1000 / 0.701 * 0.01 / 256  # length pixel in kpc

    L_box = np.arange(-25, 26)
    xx, yy, zz = np.meshgrid(L_box, L_box, L_box)
    r_p = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2) * l_pixel

    sphere_in_box = np.where(r_p <= r_halo, 1, 0)
    return sphere_in_box


def compute_spheres_in_boxes(particles, halo_masses, path):
    for particle in particles:
        halo_mass_particle = halo_masses[particle]
        s = get_sphere_in_box(halo_mass_particle)
        np.save(path + 'sphere_in_box_' + str(particle) + '.npy', s)
        del s
        gc.collect()


def ground_truth_input(halo_mass_particle, shape=(51, 51, 51)):
    return np.ones(shape) * halo_mass_particle
