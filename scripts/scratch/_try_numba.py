import numpy as np
from tensorflow.keras.utils import Sequence
import sklearn.preprocessing
import tensorflow as tf
from numba import njit, prange
import pynbody


# You should first load the snapshots and define two new keys:
# one is the 'delta' which is rho/rho_M
# and the other is 'coords' which is the coordinates of the particles in the 3D simulation grid

# ic.initial_conditions['delta'] = ic.initial_conditions['rho'].reshape((256, 256, 256))/rho_m
# k, j, i = np.unravel_index(ic.initial_conditions["iord"], (256, 256, 256))
# ic.initial_conditions['coords'] = np.column_stack((i, j, k))

class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels, sims, ic,
                 batch_size=40, dim=(51, 51, 51), n_channels=1, shuffle=False, rescale_mean=0, rescale_std=1):
        """
        :param list_IDs: this variable should be a string of the form 'sim-%i-ID-%i' % (simulation_index, particle_ID)
        :param labels: associated lables with particle IDs
        :param sims: list of simulations in the order of simulation_index
        :param batch_size: batch size
        :param dim: dimension of the sub-box to generate
        :param n_channels: number of channels in the input
        :param shuffle: this should always be False or you have a bug.. leave it as an option so I can fix later
        :param rescale_mean: mean of inputs to use of rescaling
        :param rescale_std: std of inputs to use of rescaling

        """

        self.list_IDs = list_IDs
        self.labels = labels
        self.sims = sims

        self.shuffle = shuffle
        self.dim = dim
        self.res = dim[0]
        self.batch_size = batch_size
        self.n_channels = n_channels

        self.rescale_mean = rescale_mean
        self.rescale_std = rescale_std
        rho_m = pynbody.analysis.cosmology.rho_M(ic.initial_conditions, unit="Msol kpc**-3")
        self.deltas = ic.initial_conditions['rho'].reshape((256, 256, 256))/rho_m
        k, j, i = np.unravel_index(ic.initial_conditions["iord"], (256, 256, 256))
        self.coords = ic.initial_conditions['coords'] = np.column_stack((i, j, k))

        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate a batch of data """
        indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            print("Be very aware of setting shuffle=True! I think there is a bug here")
            np.random.shuffle(self.indexes)

    def _process_input(self, s):

        # Rescale inputs
        s_t_rescaled = (s - self.rescale_mean) / self.rescale_std

        return s_t_rescaled.reshape((*self.dim, self.n_channels))

    def __data_generation(self, list_IDs_temp):
        """ Loads data containing batch_size samples """

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            sim_index = int(ID[4])
            particle_ID = int(ID[9:])
            print(particle_ID)

            sim_snapshot = self.sims[sim_index]
            i0, j0, k0 = self.coords[particle_ID]
            delta_sim = self.deltas

            output_matrix = np.zeros((self.res, self.res, self.res))
            s = slicer_numba(i0, j0, k0, self.res, delta_sim, output_matrix)

            X[i] = self._process_input(s)
            y[i] = self.labels[particle_ID]

        return X, y


@njit(parallel=True)
def slicer_numba(i0, j0, k0, width, input_matrix, output_matrix):
    i0 -= width // 2
    j0 -= width // 2
    k0 -= width // 2
    for i in prange(width):
        for j in prange(width):
            for k in prange(width):
                output_matrix[i, j, k] = input_matrix[(i + i0) % 256, (j + j0) % 256, (k + k0) % 256]
    return output_matrix


def get_halo_mass_scaler(sims, path="/lfstev/deepskies/luisals/"):
    halo_mass_sims = []
    for sim in sims:
        if sim == "0":
            halo_mass = np.load(path + "training_simulation/halo_mass_particles.npy")
        else:
            halo_mass = np.load(path + "reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")
        halo_mass_sims.append(halo_mass)
    halo_mass_sims = np.concatenate(halo_mass_sims)

    log_output = np.log10(halo_mass_sims[halo_mass_sims > 0])
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    minmax_scaler.fit(log_output.reshape(-1, 1))
    return minmax_scaler


def normalise_output(output, take_log=True):
    if take_log is True:
        log_output = np.log10(output[output > 0])
    else:
        log_output = output
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    minmax_scaler.fit(log_output.reshape(-1, 1))

    normalised_labels = np.zeros((len(output),))
    normalised_labels[output > 0] = minmax_scaler.transform(log_output.reshape(-1, 1)).flatten()
    return minmax_scaler, normalised_labels


def transform_halo_mass_to_binary_classification(halo_mass, threshold=1.8*10**12):
    labels = np.zeros((len(halo_mass), ))
    labels[halo_mass > threshold] = 1
    return labels


def normalise_distribution_to_given_variance(samples, variance,
                                             mean_samples=1.0040703121496461, std_samples=0.05050898332331686):
    samples_transformed = ((samples - mean_samples)/std_samples * np.sqrt(variance)) + mean_samples
    return samples_transformed