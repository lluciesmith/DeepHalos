import numpy as np
import time
import pynbody
import sklearn.preprocessing
from numba import njit, prange
from collections import OrderedDict
import warnings
import gc
import tensorflow as tf


class SimulationPreparation:
    def __init__(self, sim_IDs, path=""):
        """
        This class stores the simulations in dictionaries (accessible via sims_dic) and creates two new keys:
        the density contrast for each particle and the coordinates of each particle in the 3D box.

        """

        self.sims = sim_IDs
        self.path = path

        self.sims_dic = None
        self.generate_simulation_dictionary()

    def generate_simulation_dictionary(self):
        sims = self.sims
        sims_dic = {}

        for i, ID in enumerate(sims):
            snapshot_i = self.load_snapshot_from_simulation_ID(ID)
            snapshot_i = self.prepare_sim(snapshot_i, ID)
            sims_dic[ID] = snapshot_i

        self.sims_dic = sims_dic

    def load_snapshot_from_simulation_ID(self, sim_id):
        ''' Loads data using pynbody to SimSnap.'''
        if sim_id == "0":
            path1 = self.path + "training_simulation/snapshots/"
            snap_sim = pynbody.load(path1 + "ICs_z99_256_L50_gadget3.dat")

        else:
            path1 = self.path + "reseed" + sim_id + "_simulation/snapshots/"

            if sim_id == "2":
                snap_sim = pynbody.load(path1 + "IC_doub_z99_256.gadget3")
            elif sim_id in ["1", "3", "4", "5"]:
                snap_sim = pynbody.load(path1 + "IC.gadget3")
            else:
                snap_sim = pynbody.load(path1 + "IC.gadget2")
        return snap_sim

    def prepare_sim(self, snapshot, sim_id):
        ''' Computes density contrast & particle coords.'''
        snapshot.physical_units()

        if sim_id == "0":
            path1 = self.path + "training_simulation/snapshots/"
        else:
            path1 = self.path + "reseed" + sim_id + "_simulation/snapshots/"

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
        print("Computing density contrast in simulation took " + str((t1 - t0)/60) + " minutes.")

        shape_sim = int(round((snapshot["iord"].shape[0]) ** (1 / 3)))
        i, j, k = np.unravel_index(snapshot["iord"], (shape_sim, shape_sim, shape_sim))
        snapshot['coords'] = np.column_stack((i, j, k))

        del snapshot['rho'], snapshot['iord']
        gc.collect()
        return snapshot


class InputsPreparation:
    def __init__(self, sim_IDs,
                 load_ids=True, ids_filename="random_training_set.txt",
                 random_subset_each_sim=None, random_subset_all=None, log_high_mass_limit=None,
                 path="", scaler_output=None, scaler_type="standard",
                 return_rescaled_outputs=True, random_style="random", num_per_mass_bin=1000, num_bins=50,
                 shuffle=True, output_range=(-1, 1), ds_sd=1, verbose=0):
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

        self.random_style = random_style
        self.ds_sd = ds_sd
        if verbose != 0:
            print(f'random style = {self.random_style}')
            print(f'using training set seed {self.ds_sd}')
        self.random_subset = random_subset_each_sim
        self.random_subset_all = random_subset_all
        self.num_per_mass_bin = num_per_mass_bin
        self.num_bins = num_bins

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

            name = ["sim-" + str(sim_ID) + "-id-" + str(id_i) for id_i in ids_i]

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
            np.random.seed(self.ds_sd)
            ind = np.random.choice(np.arange(len(flattened_name)), self.random_subset_all, replace=False)
            flattened_name = flattened_name[ind]
            flattened_mass = flattened_mass[ind]
        
        elif self.random_style == "random2":
            if self.ds_sd == 0:
                np.random.seed(100)
            elif self.ds_sd is not None:
                np.random.seed(int(self.ds_sd-1))
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

        if self.shuffle is True:
            np.random.seed(5)

            ids_reordering = np.random.permutation(list(dict_i.keys()))
            labels_reordered = dict([(key, dict_i[key]) for key in ids_reordering])

        else:
            labels_reordered = dict_i

        self.particle_IDs = list(labels_reordered.keys())
        self.labels_particle_IDS = labels_reordered

    def get_indices_array_sampled_evenly_in_each_bin(self, array, number_bins, number_samples_per_bin):
        '''
        Split masses into num_bins and sample number_samples_per_bin
        particles randomly from each bin. Concatenate and return as single array.
        '''
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
                                     % (bins[i-1], bins[i], len(ind_bins), number_samples_per_bin))
                    ind.append(ind_bins)
                else:
                    ind_i = np.random.choice(ind_bins, number_samples_per_bin, replace=False)
                    ind.append(ind_i)
            else:
                pass

        ind = np.concatenate(ind)
        return ind

    def generate_random_set(self, simulation_ID):
        ''' Return indices and log halo mass (of all in sample or only those below
        certain log mass limit).'''
        
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
        '''Returns simulation IDs and corresponding masses (at z=0).'''
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


class DataGenerator:
    def __init__(self, list_IDs, labels, sims, batch_size=80, dim=(51, 51, 51), n_channels=1, shuffle=False, path=None, drop_remainder=True,
                 rescale_mean=0, rescale_std=1, input_type="raw", num_shells=None, dtype="float32", verbose=0, cache=True, cache_path=None):
        """
        This class creats the data generator that should be used to fit the deep learning model.
        Use DataGenerator.get_dataset() to get the dataset (batched, shuffled and prefetched) produced with tf.data.Dataset.
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
        self.num_IDs = len(list_IDs)
        self.labels = labels

        self.sims = sims
        sim_id0 = list(sims.keys())[0]
        self.shape_sim = int(round((sims[sim_id0]["iord"].shape[0]) ** (1 / 3)))

        self.shuffle = shuffle
        if verbose != 0:
            print(f'DataGenerator shuffle={self.shuffle}')
        self.dim = dim
        self.res = dim[0]
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.cache = cache
        self.prefetch = True
        self.drop_remainder = drop_remainder
        self.path = path + "inputs_avg/inp_avg" if input_type == "averaged" else path + "inputs_raw/inp_raw"
        if cache_path is None:
            cache_path = path
        self.cache_path = cache_path

        self.rescale_mean = rescale_mean
        self.rescale_std = rescale_std
        self.sims_rescaled_density = OrderedDict()
        self.preprocess_density_contrasts()

        self.input_type = input_type
        self.dtype = dtype
        warn_float_casting(self.sims_rescaled_density[sim_id0], self.dtype)

        if input_type == "averaged":
            self.num_shells = num_shells
            self.shell_labels = assign_shell_to_pixels(self.res, self.num_shells)

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def get_dataset(self):
        output_signature = tf.TensorSpec(shape=(self.res, self.res, self.res, self.n_channels), dtype=self.dtype), \
                           tf.TensorSpec(shape=(), dtype=self.dtype)
        num_threads = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        if self.shuffle is True:
            dataset = dataset.shuffle(buffer_size=5000)
        # dataset = dataset.map(self.map_generator, num_parallel_calls=num_threads)
        if self.cache is True:
            dataset = dataset.cache(self.cache_path)
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        if self.prefetch is True:
            dataset = dataset.prefetch(buffer_size=num_threads)
        return dataset

    def generator(self):
        for sample in self.list_IDs:
            yield self.get_input(sample), self.labels[sample]

    # @tf.function
    # def map_generator(self, x_elem, label_elem):
    #     x_input = tf.py_function(func=self.get_input, inp=[x_elem], Tout=self.dtype)
    #     return x_input, label_elem

    def get_input(self, ID):
        sim_index = ID[ID.find('sim-') + len('sim-'): ID.find('-id')]
        particle_ID = int(ID[ID.find('-id-') + len('-id-'):])
        inputs_file = self.load_input(sim_index, particle_ID)
        return inputs_file.reshape((*self.dim, self.n_channels))

    def generate_input(self, simulation_index, particle_id):
        i0, j0, k0 = self.sims[simulation_index]['coords'][particle_id]
        delta_sim = self.sims_rescaled_density[simulation_index]

        output_matrix = np.zeros((self.res, self.res, self.res))
        s = compute_subbox(i0, j0, k0, self.res, delta_sim, output_matrix, self.shape_sim)
        if self.input_type == "averaged":
            s = get_spherically_averaged_box(s, self.shell_labels)
        return s

    def load_input(self, simulation_index, particle_id):
        return np.load(self.path + "_sim_" + simulation_index + "_particle_" + str(particle_id) + ".npy")

    def preprocess_density_contrasts(self):
        for i, simulation in self.sims.items():
            self.sims_rescaled_density[i] = self.rescaled_density_contrast_3d(simulation)

    def rescaled_density_contrast_3d(self, sim):
        d = (sim['den_contrast'] - self.rescale_mean) / self.rescale_std
        return d.reshape(self.shape_sim, self.shape_sim, self.shape_sim)


@njit(parallel=True)
def compute_subbox(i0, j0, k0, width, input_matrix, output_matrix, shape_input):
    i0 -= width // 2
    j0 -= width // 2
    k0 -= width // 2
    for i in prange(width):
        for j in prange(width):
            for k in prange(width):
                output_matrix[i, j, k] = input_matrix[(i + i0) % shape_input, (j + j0) % shape_input, (k + k0) % shape_input]
    return output_matrix


def assign_shell_to_pixels(width, number_shells, r_shells=None):
    if r_shells is None:
        r_shells = np.linspace(2, width / 2, number_shells, endpoint=True)

    x_coord, y_coord, z_coord = np.unravel_index(np.arange(width**3), (width, width, width))
    x_coord -= width // 2
    y_coord -= width // 2
    z_coord -= width // 2
    r_coords = np.sqrt(x_coord ** 2 + y_coord ** 2 + z_coord ** 2)

    shell_labels = np.ones((width**3)) * -1
    for i in range(width**3):
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
    cumsum = np.zeros(shell_matrix.max() + 2)
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


def warn_float_casting(input_, output_dtype):
    input_dtype = input_.dtype
    message = ""
    if input_dtype != output_dtype:
        message += f"Casting from {input_dtype} to {output_dtype}. "
    if output_dtype == "float64":
        message += "WARNING: pooling will not work in float64."        
    print(message)
