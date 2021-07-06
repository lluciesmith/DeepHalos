import numpy as np
from scipy.constants import G


def get_delta_in_fourier_space(delta):
    delta_fourier = np.fft.fftn(delta.reshape(256, 256, 256))
    return delta_fourier


def get_k_coordinates_box(boxsize, path='/mnt/beegfs/home/pearl037/'):
    """ Return the k-coordinates of the simulation box """
    a = np.load(path + 'Fourier_transform_matrix.npy')
    k_coord = 2. * np.pi * a / boxsize
    return k_coord


def get_norm_k_box(boxsize, path='/mnt/beegfs/home/pearl037/'):
    k_coord = get_k_coordinates_box(boxsize, path=path)
    norm_squared = np.power(np.fabs(k_coord), 2)
    return norm_squared


def poisson_equation_in_fourier_space(delta_fourier, boxsize, path='/mnt/beegfs/home/pearl037/'):
    """ Poisson's equation in Fourier space """
    norm_k_squared = get_norm_k_box(boxsize, path=path)
    norm_k_squared[0, 0, 0] = 1
    phi = - delta_fourier / norm_k_squared
    phi[0, 0, 0] = 0
    return phi


def get_potential_from_density(delta, boxsize, path='/mnt/beegfs/home/pearl037/'):
    """
    Computes the gravitational potential give the density via Poisson's equation in Fourier space.
    """
    delta_fourier = get_delta_in_fourier_space(delta)
    potential_fourier = poisson_equation_in_fourier_space(delta_fourier, boxsize, path=path)
    potential_real = np.real(np.fft.ifftn(potential_fourier)).flatten()
    return potential_real