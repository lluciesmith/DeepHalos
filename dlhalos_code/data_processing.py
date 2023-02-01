# You should first load the snapshots and define two new keys:
# one is the 'delta' which is rho/rho_M
# and the other is 'coords' which is the coordinates of the particles in the 3D simulation grid

# This module takes care of the data preparation: it generates the DataGenerator used in the model.

import numpy as np
import time
import pynbody
import sklearn.preprocessing
from numba import njit, prange
from tensorflow.keras.utils import Sequence
from collections import OrderedDict
import threading
import warnings
import gc
from dlhalos_code import potential as pot


class SimulationPreparation:
    def __init__(self, sim_IDs, potential=False, path="/lfstev/deepskies/luisals/"):
        """
        This class stores the simulations in dictionaries (accessible via sims_dic) and creates two new keys:
        the density contrast for each particle and the coordinates of each particle in the 3D box.

        """

        self.sims = sim_IDs
        self.path = path
        self.potential = potential

        self.sims_dic = None
        self.generate_simulation_dictionary()

    def generate_simulation_dictionary(self):
        sims = self.sims
        sims_dic = {}

        for i, ID in enumerate(sims):
            snapshot_i = self.load_snapshot_from_simulation_ID(ID)
            snapshot_i = self.prepare_sim(snapshot_i, ID, potential=self.potential)
            sims_dic[ID] = snapshot_i

        self.sims_dic = sims_dic

    def load_snapshot_from_simulation_ID(self, sim_id):
        if sim_id == "0":
            path1 = self.path + "training_simulation/snapshots/"
            # path1 = self.path
            # path1 = "/Users/lls/Documents/mlhalos_files/Nina-Simulations/double/"
            snap_sim = pynbody.load(path1 + "ICs_z99_256_L50_gadget3.dat")

        elif len(sim_id) == 2 or len(sim_id) == 1:
            path1 = self.path + "reseed" + sim_id + "_simulation/snapshots/"
            # path1 = "/Users/lls/Documents/mlhalos_files/reseed50/"
            # path1 = "/Users/lls/Documents/mlhalos_files/reseed6/"

            if sim_id == "2":
                snap_sim = pynbody.load(path1 + "IC_doub_z99_256.gadget3")
            elif sim_id in ["1", "3", "4", "5"]:
                snap_sim = pynbody.load(path1 + "IC.gadget3")
            else:
                snap_sim = pynbody.load(path1 + "IC.gadget2")
        else:
            snap_sim = pynbody.load(self.path + sim_id + "/snapshots/IC.gadget3")
        return snap_sim

    def prepare_sim(self, snapshot, sim_id, potential=False):
        snapshot.physical_units()

        if sim_id == "0":
            path1 = self.path + "training_simulation/snapshots/"
            # path1 = self.path
        elif len(sim_id) == 2 or len(sim_id) == 1:
            path1 = self.path + "reseed" + sim_id + "_simulation/snapshots/"
        else:
            path1 = self.path + sim_id + "/snapshots/"

        t0 = time.time()
        try:
            rho = np.load(path1 + "density_Msol_kpc3_ics.npy")
            print("Loaded density array of simulation ID " + sim_id)
            snapshot["rho"] = rho
            snapshot["rho"].simulation = snapshot
            snapshot["rho"].units = "Msol kpc**-3"

        except FileNotFoundError:
            np.save(path1 + "density_Msol_kpc3_ics.npy", snapshot["rho"])
            print("Saved density array of simulation ID " + sim_id)

        rho_m = pynbody.analysis.cosmology.rho_M(snapshot, unit=snapshot["rho"].units)
        snapshot['den_contrast'] = snapshot["rho"] / rho_m
        t1 = time.time()
        print("Computing density contrast in simulation took " + str((t1 - t0) / 60) + " minutes.")

        shape_sim = int(round((snapshot["iord"].shape[0]) ** (1 / 3)))
        i, j, k = np.unravel_index(snapshot["iord"], (shape_sim, shape_sim, shape_sim))
        snapshot['coords'] = np.column_stack((i, j, k))

        if potential:
            delta = snapshot["rho"] / np.mean(snapshot["rho"]) - 1
            boxsize = snapshot.properties["boxsize"].in_units(snapshot["pos"].units)
            pot_field = pot.get_potential_from_density(delta, boxsize)
            snapshot["potential"] = pot_field

        del snapshot['rho'], snapshot['iord']
        gc.collect()
        return snapshot


class InputsPreparation:
    def __init__(self, sim_IDs,
                 load_ids=True, ids_filename="random_training_set.txt",
                 random_subset_each_sim=None, random_subset_all=None, log_high_mass_limit=None, weights=False,
                 path="/lfstev/deepskies/luisals/", scaler_output=None, scaler_type="standard",
                 return_rescaled_outputs=True, random_style="random", num_per_mass_bin=1000, num_bins=50,
                 shuffle=True, output_range=(-1, 1)):
        """
        This class prepares the inputs in the correct format for the DataGenerator class.
        Particles and their labels are stored in a dictionary s.t.  particles are identified via the string
        'sim-#-id-*' where # is the sim ID and * is the particle ID.

        Particles can be loaded from an existing file or randomly picked from the simulation.

        """
        self.sims = sim_IDs

        self.load_ids = load_ids
        self.ids_filename = ids_filename
        self.path = path
        self.log_high_mass_limit = log_high_mass_limit
        self.shuffle = shuffle
        self.weights = weights

        # How to sample particles for the training/validation/test set

        self.random_style = random_style
        self.random_subset = random_subset_each_sim
        self.random_subset_all = random_subset_all
        self.num_per_mass_bin = num_per_mass_bin
        self.num_bins = num_bins

        # How to rescale the output

        self.return_rescaled_outputs = return_rescaled_outputs
        self.scaler_output = scaler_output
        self.scaler_type = scaler_type
        self.output_range = output_range

        self.particle_IDs = None
        self.labels_particle_IDS = None
        self.generate_particle_IDs_dictionary()

    def generate_particle_IDs_dictionary(self):
        names = []
        masses = []
        for i, sim_ID in enumerate(self.sims):
            if self.load_ids:
                ids_i, mass_i = self.load_ids_from_file(sim_ID)
            else:
                ids_i, mass_i = self.generate_random_set(sim_ID)

            name = ['sim-' + str(sim_ID) + '-id-' + str(id_i) for id_i in ids_i]
            names.append(name)
            masses.append(mass_i)

        flattened_name = np.concatenate(names)
        flattened_mass = np.concatenate(masses)

        if self.random_style == "uniform":
            indices = self.get_indices_array_sampled_evenly_in_each_bin(flattened_mass, self.num_bins,
                                                                        self.num_per_mass_bin)
            flattened_name = flattened_name[indices]
            flattened_mass = flattened_mass[indices]

        elif self.random_style == "random":
            ind = np.random.choice(np.arange(len(flattened_name)), self.random_subset_all, replace=False)
            flattened_name = flattened_name[ind]
            flattened_mass = flattened_mass[ind]

        else:
            pass

        if self.return_rescaled_outputs is True:
            if self.scaler_output is None:
                output_ids, self.scaler_output = self.fit_scaler_and_transform(flattened_mass)
            else:
                output_ids = self.transform_array_given_scaler(self.scaler_output, flattened_mass)
        else:
            output_ids = flattened_mass

        dict_i = OrderedDict(zip(flattened_name, output_ids))

        if self.weights is True:
            weights_particle_IDs = self.get_weights_samples(flattened_mass)
            dict_weights = OrderedDict(zip(flattened_name, weights_particle_IDs))

            if self.shuffle is True:
                np.random.seed(5)

                ids_reordering = np.random.permutation(list(dict_i.keys()))
                labels_reordered = dict([(key, dict_i[key]) for key in ids_reordering])
                weights_reordered = dict([(key, dict_weights[key]) for key in ids_reordering])
            else:
                labels_reordered = dict_i
                weights_reordered = dict_weights

            self.particle_IDs = list(labels_reordered.keys())
            self.labels_particle_IDS = labels_reordered
            self.weights_particle_IDs = weights_reordered

        else:

            if self.shuffle is True:
                np.random.seed(5)

                ids_reordering = np.random.permutation(list(dict_i.keys()))
                labels_reordered = dict([(key, dict_i[key]) for key in ids_reordering])

            else:
                labels_reordered = dict_i

            self.particle_IDs = list(labels_reordered.keys())
            self.labels_particle_IDS = labels_reordered

    def get_indices_array_sampled_evenly_in_each_bin(self, array, number_bins, number_samples_per_bin):
        bins = np.histogram_bin_edges(array, bins=number_bins)
        ind = []

        for i in np.arange(1, len(bins)):
            if i == len(bins) - 1:
                ind_bins = np.where((array >= bins[i - 1]) & (array <= bins[i]))[0]
            else:
                ind_bins = np.where((array >= bins[i - 1]) & (array < bins[i]))[0]

            if ind_bins.size != 0:
                if len(ind_bins) <= number_samples_per_bin:
                    warnings.warn("The number of particles in bin [%.3f, %.3f) is %i. This is smaller than the "
                                  "requested %i number of particles. We include all particles in this bin."
                                  % (bins[i - 1], bins[i], len(ind_bins), number_samples_per_bin))
                    ind.append(ind_bins)
                else:
                    ind_i = np.random.choice(ind_bins, number_samples_per_bin, replace=False)
                    ind.append(ind_i)
            else:
                pass

        ind = np.concatenate(ind)
        return ind

    def generate_random_set(self, simulation_ID):
        if simulation_ID == "0":
            halo_mass = np.load(self.path + "training_simulation/halo_mass_particles.npy")
        else:
            halo_mass = np.load(
                self.path + "reseed" + simulation_ID + "_simulation/reseed" + simulation_ID + "_halo_mass_particles.npy")
        ids_in_halo = np.where(halo_mass > 0)[0]

        if self.log_high_mass_limit is not None:
            ind = np.log10(halo_mass[ids_in_halo]) <= self.log_high_mass_limit
            ids_in_halo = ids_in_halo[ind]

        if self.random_subset is not None:
            ids_i = np.random.choice(ids_in_halo, self.random_subset, replace=False)
            return ids_i, np.log10(halo_mass[ids_i])
        else:
            return ids_in_halo, np.log10(halo_mass[ids_in_halo])

    def load_ids_from_file(self, simulation_ID):
        ids_i, mass_i = self.get_ids_and_regression_labels(sim=simulation_ID)
        return ids_i, mass_i

    def get_ids_and_regression_labels(self, sim="0"):
        if sim == "0":
            path1 = self.path + "training_simulation/training_sim_"
            halo_mass = np.load(self.path + "training_simulation/halo_mass_particles.npy")
        else:
            path1 = self.path + "reseed" + sim + "_simulation/reseed_" + sim + "_"
            halo_mass = np.load(self.path + "reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")

        with open(path1 + self.ids_filename, "r") as f:
            ids_bc = np.array([line.rstrip("\n") for line in f]).astype("int")

        if self.log_high_mass_limit is not None:
            ind = np.log10(halo_mass[ids_bc]) <= self.log_high_mass_limit
            ids_bc = ids_bc[ind]

        if self.random_subset is not None:
            ids_i = np.random.choice(ids_bc, self.random_subset, replace=False)
            ids_bc = ids_i

        output_ids = np.log10(halo_mass[ids_bc])
        return ids_bc, output_ids

    def fit_scaler_and_transform(self, array_outputs):
        if self.scaler_type == "standard":
            norm_scaler = sklearn.preprocessing.StandardScaler()
        elif self.scaler_type == "minmax":
            norm_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=self.output_range)
        else:
            raise NameError("Choose between 'standard' and 'minmax' scalers")

        norm_scaler.fit(array_outputs.reshape(-1, 1))

        rescaled_out = self.transform_array_given_scaler(norm_scaler, array_outputs)
        return rescaled_out, norm_scaler

    def transform_array_given_scaler(self, scaler, array):
        scaled_array = scaler.transform(array.reshape(-1, 1)).flatten()
        return scaled_array

    def get_weights_samples(self, outputs):
        bins = np.load("/lfstev/deepskies/luisals/regression/large_CNN/weighted/log_m_Msol_bins.npy")
        weights = np.load("/lfstev/deepskies/luisals/regression/large_CNN/weighted/weights.npy")

        weights_samples = np.ones((len(outputs),))
        for i in range(len(bins) - 1):
            if i == range(len(bins) - 1)[-1]:
                ind = np.where((outputs >= bins[i]) & (outputs <= bins[i + 1]))[0]
            else:
                ind = np.where((outputs >= bins[i]) & (outputs < bins[i + 1]))[0]

            weights_samples[ind] = weights[i]

        return weights_samples


class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, sims, weights=None,
                 batch_size=80, dim=(51, 51, 51), n_channels=1, shuffle=False,
                 rescale_mean=0, rescale_std=1,
                 input_type="raw", num_shells=None, path=None):
        """
        This class created the data generator that should be used to fit the deep learning model.

        :param list_IDs: this variable should be a string of the form 'sim-%i-ID-%i' % (simulation_index, particle_ID)
        :param labels: This is a dictionary of the form {particle ID: labels}
        :param sims: list of simulation IDs
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
        sim_id0 = list(sims.keys())[0]
        self.shape_sim = int(round((sims[sim_id0]["iord"].shape[0]) ** (1 / 3)))

        self.weights = weights
        self.shuffle = shuffle
        self.dim = dim
        self.res = dim[0]
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.path = path

        self.rescale_mean = rescale_mean
        self.rescale_std = rescale_std
        if input_type == "potential":
            self.sims_potential = OrderedDict()
            self.preprocess_potential()
        else:
            self.sims_rescaled_density = OrderedDict()
            self.preprocess_density_contrasts()

        self.input_type = input_type

        if input_type == "averaged":
            self.num_shells = num_shells
            self.shell_labels = assign_shell_to_pixels(self.res, self.num_shells)

        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate a batch of data """

        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        if list_IDs_temp:
            if self.weights is None:
                X, y = self.__data_generation(list_IDs_temp)
                return X, y
            else:
                X, y, w = self.__data_generation_w_weights(list_IDs_temp)
                return X, y, w
        else:
            raise IndexError("Batch " + str(index) + " is empty.")

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """ Loads data containing batch_size samples """

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            sim_index = ID[ID.find('sim-') + len('sim-'): ID.find('-id')]
            particle_ID = int(ID[ID.find('-id-') + len('-id-'):])
            s = self.load_input(sim_index, particle_ID)
            #s = self.generate_input(sim_index, particle_ID)

            X[i] = self._process_input(s)
            y[i] = self.labels[ID]

        return X, y

    def __data_generation_w_weights(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,))
        w = np.empty((self.batch_size,))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            sim_index = ID[ID.find('sim-') + len('sim-'): ID.find('-id')]

            # generate box
            particle_ID = int(ID[ID.find('-id-') + + len('-id-'):])
            s = self.load_input(sim_index, particle_ID)
            #s = self.generate_input(sim_index, particle_ID)

            X[i] = self._process_input(s)
            y[i] = self.labels[ID]
            w[i] = self.weights[ID]

        return X, y, w

    def _process_input(self, s):
        # s_t = np.transpose(s, axes=(1, 0, 2))
        # Rescale inputs
        # s_t_rescaled = (s_t - self.rescale_mean) / self.rescale_std
        return s.reshape((*self.dim, self.n_channels))

    def load_input(self, simulation_index, particle_id):
        path = self.path + "inputs_avg/inp_avg" if self.input_type == "averaged" else self.path + "inputs_raw/inp_raw"
        return np.load(path + "_sim_" + simulation_index + "_particle_" + particle_id + ".npy")

    def generate_input(self, simulation_index, particle_id):
        i0, j0, k0 = self.sims[simulation_index]['coords'][particle_id]
        if self.input_type == "potential":
            delta_sim = self.sims_potential[simulation_index]
        else:
            delta_sim = self.sims_rescaled_density[simulation_index]

        output_matrix = np.zeros((self.res, self.res, self.res))
        s = compute_subbox(i0, j0, k0, self.res, delta_sim, output_matrix, self.shape_sim)

        if self.input_type == "averaged":
            s = get_spherically_averaged_box(s, self.shell_labels)

        return s

    def preprocess_potential(self):
        for i, simulation in self.sims.items():
            self.sims_potential[i] = self.rescaled_qty_3d(simulation, qty="potential")

    def preprocess_density_contrasts(self):
        for i, simulation in self.sims.items():
            self.sims_rescaled_density[i] = self.rescaled_qty_3d(simulation, qty="den_contrast")

    def rescaled_qty_3d(self, sim, qty="den_contrast"):
        d = (sim[qty] - self.rescale_mean) / self.rescale_std
        return d.reshape(self.shape_sim, self.shape_sim, self.shape_sim)


@njit(parallel=True)
def compute_subbox(i0, j0, k0, width, input_matrix, output_matrix, shape_input):
    i0 -= width // 2
    j0 -= width // 2
    k0 -= width // 2
    for i in prange(width):
        for j in prange(width):
            for k in prange(width):
                output_matrix[i, j, k] = input_matrix[
                    (i + i0) % shape_input, (j + j0) % shape_input, (k + k0) % shape_input]
    return output_matrix


def assign_shell_to_pixels(width, number_shells, r_shells=None):
    if r_shells is None:
        r_shells = np.linspace(2, width / 2, number_shells, endpoint=True)

    x_coord, y_coord, z_coord = np.unravel_index(np.arange(width ** 3), (width, width, width))
    x_coord -= width // 2
    y_coord -= width // 2
    z_coord -= width // 2
    r_coords = np.sqrt(x_coord ** 2 + y_coord ** 2 + z_coord ** 2)

    shell_labels = np.ones((width ** 3)) * -1
    for i in range(width ** 3):
        shell_beloning = np.where(r_coords[i] <= r_shells)[0]

        if shell_beloning.size != 0:
            shell_labels[i] = shell_beloning.min()

    shell_labels = shell_labels.reshape(width, width, width)
    return shell_labels.astype("int")


def get_spherically_averaged_box_slow(input_matrix, shell_matrix):
    averaged_box = np.zeros_like(input_matrix)
    shell_labels = np.unique(shell_matrix[shell_matrix >= 0])

    for shell_index in shell_labels:
        averaged_box[shell_matrix == shell_index] = np.mean(input_matrix[shell_matrix == shell_index])
    return averaged_box


@njit(parallel=True)
def _get_spherically_averaged_box_w_numba(input_matrix, shell_matrix):
    cumsum = np.zeros(shell_matrix.max() + 2)  # the last index will match for shell_index == -1
    counts = np.zeros_like(cumsum)
    for shell_index, input_value in zip(shell_matrix.flatten(), input_matrix.flatten()):
        if shell_index < 0:
            cumsum[shell_index] = 0
        else:
            cumsum[shell_index] += input_value
        counts[shell_index] += 1
    mean_per_shell = cumsum / counts
    return mean_per_shell


def get_spherically_averaged_box(input_matrix, shell_matrix):
    return _get_spherically_averaged_box_w_numba(input_matrix, shell_matrix)[shell_matrix]





