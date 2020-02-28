# You should first load the snapshots and define two new keys:
# one is the 'delta' which is rho/rho_M
# and the other is 'coords' which is the coordinates of the particles in the 3D simulation grid

# Then prepare the data in the right format and generate the DataGenerator

import numpy as np
import time
import pynbody
import sklearn.preprocessing
from numba import njit, prange
from tensorflow.keras.utils import Sequence


class SimulationPreparation:
    def __init__(self, sim_IDs, path="/lfstev/deepskies/luisals/"):
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
            snapshot_i = self.prepare_sim(snapshot_i)
            sims_dic[ID] = snapshot_i

        self.sims_dic = sims_dic

    def load_snapshot_from_simulation_ID(self, sim_id):
        if sim_id == "0":
            path1 = self.path + "training_simulation/snapshots/"
            # path1 = "/Users/lls/Documents/mlhalos_files/Nina-Simulations/double/"
            snap_sim = pynbody.load(path1 + "ICs_z99_256_L50_gadget3.dat")

        else:
            path1 = self.path + "reseed" + sim_id + "_simulation/snapshots/"
            # path1 = "/Users/lls/Documents/mlhalos_files/reseed50/"

            if sim_id == "2":
                snap_sim = pynbody.load(path1 + "IC_doub_z99_256.gadget3")
            elif sim_id in ["1", "3", "4", "5"]:
                snap_sim = pynbody.load(path1 + "IC.gadget3")
            else:
                snap_sim = pynbody.load(path1 + "IC.gadget2")
        return snap_sim

    def prepare_sim(self, snapshot):
        snapshot.physical_units()

        t0 = time.time()
        rho_m = pynbody.analysis.cosmology.rho_M(snapshot, unit=snapshot["rho"].units)
        snapshot['den_contrast'] = snapshot['rho'] / rho_m
        t1 = time.time()
        print("Loading density contrast in simulation took " + str((t1 - t0)/60))

        shape_sim = int(round((snapshot["iord"].shape[0]) ** (1 / 3)))
        i, j, k = np.unravel_index(snapshot["iord"], (shape_sim, shape_sim, shape_sim))
        snapshot['coords'] = np.column_stack((i, j, k))
        return snapshot


class InputsPreparation:
    def __init__(self, sim_IDs, load_ids=True, ids_filename="random_training_set.txt", random_subset_each_sim=None,
                 path="/lfstev/deepskies/luisals/", scaler_output=None, return_rescaled_outputs=True, shuffle=True):
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
        self.random_subset = random_subset_each_sim
        self.return_rescaled_outputs = return_rescaled_outputs
        self.shuffle = shuffle

        self.scaler_output = scaler_output
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

        if self.return_rescaled_outputs is True:
            if self.scaler_output is None:
                output_ids, self.scaler_output = self.get_standard_scaler_and_transform(flattened_mass)
            else:
                output_ids = self.transform_array_given_scaler(self.scaler_output, flattened_mass)
        else:
            output_ids = flattened_mass

        dict_i = dict(zip(flattened_name, output_ids))

        if self.shuffle is True:
            np.random.seed(5)
            ids_reordering = np.random.permutation(list(dict_i.keys()))
            labels_reordered = dict([(key, dict_i[key]) for key in ids_reordering])
        else:
            labels_reordered = dict_i

        self.particle_IDs = list(labels_reordered.keys())
        self.labels_particle_IDS = labels_reordered

    def generate_random_set(self, simulation_ID):
        if simulation_ID == "0":
            # halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
            halo_mass = np.load(self.path + "training_simulation/halo_mass_particles.npy")
        else:
            # halo_mass = np.load("/Users/lls/Documents/mlhalos_files/reseed50/features/halo_mass_particles.npy")
            halo_mass = np.load(self.path + "reseed" + simulation_ID + "_simulation/reseed" + simulation_ID +
                                "_halo_mass_particles.npy")

        ids_in_halo = np.where(halo_mass > 0)[0]
        ids_i = np.random.choice(ids_in_halo, self.random_subset, replace=False)
        mass_i = np.log10(halo_mass[ids_i])
        return ids_i, mass_i

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
        # if sim == "0":
        #     path1 = "/Users/lls/Documents/mlhalos_files/training_sim_"
        #     halo_mass = np.load("/Users/lls/Documents/mlhalos_files/stored_files/halo_mass_particles.npy")
        # else:
        #     path1 = "/Users/lls/Documents/mlhalos_files/reseed50/CNN_results/reseed_1_"
        #     halo_mass = np.load("/Users/lls/Documents/mlhalos_files/reseed50/features/halo_mass_particles.npy")

        with open(path1 + self.ids_filename, "r") as f:
            ids_bc = np.array([line.rstrip("\n") for line in f]).astype("int")
        
        if self.random_subset is not None:
            ids_i = np.random.choice(ids_bc, self.random_subset, replace=False)
            ids_bc = ids_i

        output_ids = np.log10(halo_mass[ids_bc])
        return ids_bc, output_ids

    def get_standard_scaler_and_transform(self, array_outputs):
        norm_scaler = sklearn.preprocessing.StandardScaler()
        norm_scaler.fit(array_outputs.reshape(-1, 1))

        rescaled_out = self.transform_array_given_scaler(norm_scaler, array_outputs)
        return rescaled_out, norm_scaler

    def transform_array_given_scaler(self, scaler, array):
        scaled_array = scaler.transform(array.reshape(-1, 1)).flatten()
        return scaled_array


class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, sims,
                 batch_size=80, dim=(51, 51, 51), n_channels=1, shuffle=False,
                 rescale_mean=0, rescale_std=1):
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

        self.shuffle = shuffle
        self.dim = dim
        self.res = dim[0]
        self.batch_size = batch_size
        self.n_channels = n_channels

        self.rescale_mean = rescale_mean
        self.rescale_std = rescale_std

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
        s_t = np.transpose(s, axes=(1, 0, 2))
        # Rescale inputs
        s_t_rescaled = (s_t - self.rescale_mean) / self.rescale_std

        return s_t_rescaled.reshape((*self.dim, self.n_channels))

    def __data_generation(self, list_IDs_temp):
        """ Loads data containing batch_size samples """

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            sim_index = ID[4]

            # generate box

            particle_ID = int(ID[9:])

            sim_snapshot = self.sims[sim_index]
            i0, j0, k0 = sim_snapshot['coords'][particle_ID]
            delta_sim = sim_snapshot['den_contrast'].reshape(self.shape_sim, self.shape_sim, self.shape_sim)

            output_matrix = np.zeros((self.res, self.res, self.res))
            s = compute_subbox(i0, j0, k0, self.res, delta_sim, output_matrix, self.shape_sim)

            # load box

            # particle_ID = ID[9:]
            #
            # if self.res == 51:
            #     path_midddle = "training_set/"
            # elif self.res == 75:
            #     path_midddle = "training_set_res75/"
            # else:
            #     raise (ValueError, "I have subboxes only for 51 or 75 cubed resolution.")
            #
            # if sim_index == "0":
            #     s = np.load('/lfstev/deepskies/luisals/training_simulation/' + path_midddle + particle_ID +
            #                 '/subbox_' + str(self.res) + '_particle_' + particle_ID + '.npy')
            # else:
            #     s = np.load("/lfstev/deepskies/luisals/reseed" + sim_index + "_simulation/" + path_midddle +
            #                 particle_ID + '/subbox_' + str(self.res) + '_particle_' + particle_ID + '.npy')

            X[i] = self._process_input(s)
            y[i] = self.labels[ID]

        return X, y


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

