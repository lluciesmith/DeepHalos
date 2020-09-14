import numpy as np
import pynbody
import dlhalos_code.data_processing as dp
from tensorflow.keras.utils import Sequence
from collections import OrderedDict
import time


class DataGenerator_z0(Sequence):
    def __init__(self, list_IDs, labels, sims, res_sim=1667, rescale=False, gridded_box=None,
                 batch_size=80, dim=(75, 75, 75), n_channels=1, shuffle=False):
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
        self.res_sim = res_sim

        self.shuffle = shuffle
        self.dim = dim
        self.res = dim[0]

        self.batch_size = batch_size
        self.n_channels = n_channels

        # For every simulation, compute the gridded density at z=0
        if gridded_box is None:
            self.box_class = OrderedDict()
            self.preprocess_gridded_densities(rescale=rescale)
        else:
            self.box_class = gridded_box

        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate a batch of data """

        indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        if list_IDs_temp:
            X, y = self.__data_generation(list_IDs_temp)
            return X, y
        else:
            raise IndexError("Batch " + str(index) + " is empty.")

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            print("Be very aware of setting shuffle=True! I think there is a bug here")
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """ Loads data containing batch_size samples """

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            sim_index = ID[ID.find('sim-') + 4: ID.find('-id')]

            # generate box

            particle_ID = int(ID[ID.find('-id-') + 4:])
            s = self.generate_input(sim_index, particle_ID)

            X[i] = s.reshape((*self.dim, self.n_channels))
            y[i] = self.labels[ID]

        return X, y

    def preprocess_gridded_densities(self, rescale=False):
        for i, simulation in self.sims.items():
            self.box_class[i] = Boxz0(i, simulation, self.res_sim, rescale=rescale)

    def generate_input(self, simulation_index, particle_id):
        class_sim = self.box_class[simulation_index]
        z0_box = class_sim.generate_z0_box(particle_id, self.res)
        return z0_box


class Boxz0:
    def __init__(self, sim_id, snapshot, res_box, rescale=False, path="/lfstev/deepskies/luisals/"):
        # if not np.allclose(snapshot["iord"], np.arange(len(snapshot))):
        #     raise ValueError("The snapshot properties are not ordered by particle ID so this code will break")

        if sim_id == "0":
            self.path1 = path + "training_simulation/snapshots/"
        else:
            self.path1 = path + "reseed" + sim_id + "_simulation/snapshots/"

        boxsize = float(snapshot.properties['boxsize'].in_units(snapshot['pos'].units))
        grid_spacing = boxsize/res_box

        self.sim_id = sim_id
        self.snapshot = snapshot
        self.res_box = res_box
        self.grid_spacing = grid_spacing

        self.coords_grid = self.get_grid_label(snapshot, grid_spacing)
        self.rescaled_rho_grid = self.get_rescaled_density_in_3d_grid(snapshot, res_box, rescale=rescale)

    def get_grid_label(self, snapshot, grid_spacing):
        # Make sure the grid spacing is in Mpc h**-1
        snapshot.physical_units(distance="Mpc h**-1")

        pos_particles = snapshot["pos"]
        grid_coords = (pos_particles/grid_spacing).astype('int')
        assert grid_coords.min() == 0
        return grid_coords

    def deposit_density_on_grid(self, snapshot, res_box):
        try:
            rho_grid = np.load(self.path1 + "z0_log_density_constrast_on_grid_" + str(res_box) + ".npy")
            print("Loaded z=0 gridded density array of simulation ID " + self.sim_id)

        except FileNotFoundError:
            rho_grid = pynbody.sph.to_3d_grid(snapshot, qty="log_den_contrast", nx=res_box, threaded=True)
            np.save(self.path1 + "z0_log_density_constrast_on_grid_" + str(res_box) + ".npy", rho_grid)
            print("Saved z=0 gridded density array of simulation ID " + self.sim_id)

        return rho_grid

    def rescale_density_on_grid(self, rho_gridded):
        mean = np.mean(rho_gridded)
        print("Mean of the gridded log density contrast is " + str(mean))
        std = np.std(rho_gridded)
        print("Standard deviation of the gridded log density contrast is " + str(std))

        d = (rho_gridded - mean) / std
        return d

    def get_rescaled_density_in_3d_grid(self, snapshot, res_box, rescale=False):
        r = self.deposit_density_on_grid(snapshot, res_box)
        if rescale is False:
            return r
        else:
            r_rescaled = self.rescale_density_on_grid(r)
            return r_rescaled

    def generate_z0_box(self, particle_id, width_subbox):

        # CAREFUL HERE FOR SIMULATIONS WHERE THE POSITIONS ARE NOT ORDERED BY PARTICLE ID
        ind = int(np.where(self.snapshot['iord'] == particle_id)[0])
        i0, j0, k0 = self.coords_grid[ind]

        output_matrix = np.zeros((width_subbox, width_subbox, width_subbox))
        s = dp.compute_subbox(i0, j0, k0, width_subbox, self.rescaled_rho_grid, output_matrix, self.res_box)

        return s


class SimulationPreparation_z0:
    def __init__(self, sim_IDs, path="/lfstev/deepskies/luisals/"):
        """
        This class stores the simulations in dictionaries (accessible via sims_dic) and creates one new key:
        the log of the density contrast for each particle

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
        if sim_id == "0":
            path1 = self.path + "training_simulation/snapshots/"
            #path1 = self.path
            snap_sim = pynbody.load(path1 + "snapshot_104")

        else:
            path1 = self.path + "reseed" + sim_id + "_simulation/snapshots/"

            if sim_id in ["1", "2", "3", "4", "5"]:
                snap_sim = pynbody.load(path1 + "snapshot_099")
            else:
                snap_sim = pynbody.load(path1 + "snapshot_007")
        return snap_sim

    def prepare_sim(self, snapshot, sim_id):
        snapshot.physical_units(distance="Mpc h**-1")

        if sim_id == "0":
            path1 = self.path + "training_simulation/snapshots/"
            # path1 = self.path
        else:
            path1 = self.path + "reseed" + sim_id + "_simulation/snapshots/"

        t0 = time.time()
        try:
            rho = np.load(path1 + "density_Msol_Mpc-3_h3_z0.npy")
            print("Loaded z=0 density array of simulation ID " + sim_id)
            snapshot["rho"] = rho
            snapshot["rho"].simulation = snapshot
            snapshot["rho"].units = "h**3 Msol Mpc**-3"

        except FileNotFoundError:
            np.save(path1 + "density_Msol_Mpc-3_h3_z0.npy", snapshot["rho"])
            print("Saved density array of simulation ID " + sim_id)

        rho_m = pynbody.analysis.cosmology.rho_M(snapshot, unit=snapshot["rho"].units)
        den_con = snapshot["rho"] / rho_m
        snapshot["log_den_contrast"] = np.log10(den_con)

        t1 = time.time()
        print("Computing density contrast (z=0) in simulation took " + str((t1 - t0)/60) + " minutes.")

        return snapshot