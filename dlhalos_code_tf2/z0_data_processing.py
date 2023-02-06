import numpy as np
import pynbody
#pynbody.config['number_of_threads'] = 16
import dlhalos_code_tf2.data_processing as dp
import tensorflow as tf
from collections import OrderedDict
import time
import os


class DataGenerator_z0:
    def __init__(self, list_IDs, labels, sims, res_sim=1667, rescale=False, gridded_box=None,
                 batch_size=80, dim=(75, 75, 75), n_channels=1, shuffle=False, dtype="float32", path=""):
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
        self.num_IDs = len(list_IDs)
        self.labels = labels

        self.sims = sims
        sim_id0 = list(sims.keys())[0]
        self.res_sim = res_sim

        self.shuffle = shuffle
        self.dim = dim
        self.res = dim[0]

        self.batch_size = batch_size
        self.n_channels = n_channels
        
        self.path = path

        # For every simulation, compute the gridded density at z=0
        if gridded_box is None:
            self.box_class = OrderedDict()
            self.preprocess_gridded_densities(rescale=rescale)
        else:
            self.box_class = gridded_box
            
        self.dtype = dtype        
        dp.warn_float_casting(self.box_class[sim_id0].rescaled_rho_grid, self.dtype)

    def generate_data(self, idx):
        idx = int(idx)
        ID = self.list_IDs[idx]
        sim_index = ID[ID.find('sim-') + 4: ID.find('-id')]
        particle_ID = int(ID[ID.find('-id-') + 4:])
        s = self.generate_input(sim_index, particle_ID)
        box = s.reshape((*self.dim, self.n_channels))
        boxlabel = self.labels[ID]
        return box, boxlabel

    def get_dataset(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(tf.range(self.num_IDs))
        if self.shuffle is True:
            dataset = dataset.shuffle(self.num_IDs)
        if self.dtype == "float64":            
            Tout_dtype = tf.float64
        else:
            Tout_dtype = tf.float32
        dataset = dataset.map(lambda x: tf.py_function(func=self.generate_data, inp=[x], Tout=((Tout_dtype, Tout_dtype))),
                              num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def preprocess_gridded_densities(self, rescale=False):
        for i, simulation in self.sims.items():
            if i == "0":
                path_den = self.path + "training_simulation/snapshots/"
            else:
                path_den = self.path + "reseed" + str(i) + "_simulation/snapshots/"
            self.box_class[i] = Boxz0(i, simulation, self.res_sim, rescale=rescale, path=path_den)

    def generate_input(self, simulation_index, particle_id):
        class_sim = self.box_class[simulation_index]
        z0_box = class_sim.generate_z0_box(particle_id, self.res)
        return z0_box


class Boxz0:
    def __init__(self, sim_id, snapshot, res_box, rescale=False, path="/mnt/beegfs/work/ati/pearl037/"):
        boxsize = float(snapshot.properties['boxsize'].in_units(snapshot['pos'].units))
        grid_spacing = boxsize/res_box

        self.sim_id = sim_id
        self.snapshot = snapshot
        self.res_box = res_box
        self.grid_spacing = grid_spacing
        self.path = path

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
            rho_grid = np.load(self.path + "z0_log_density_constrast_on_grid_" + str(res_box) + ".npy")
            print("Loaded z=0 gridded density array of simulation ID " + self.sim_id)

        except FileNotFoundError:
            rho_grid = pynbody.sph.to_3d_grid(snapshot, qty="log_den_contrast", nx=res_box, threaded=True)
            np.save(self.path + "z0_log_density_constrast_on_grid_" + str(res_box) + ".npy", rho_grid)
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
    def __init__(self, params, path="/lfstev/deepskies/luisals/"):
        """
        This class stores the simulations in dictionaries (accessible via sims_dic) and creates one new key:
        the log of the density contrast for each particle

        """

        self.sims = params.sim_ids
        self.sims_snapnum = dict(zip(params.sim_ids, params.snapnum))
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
        path1 = self.path + "L50_N512_" + sim_id
        try:
            snap_sim = pynbody.load(path1 + "/snapshot_" + self.sims_snapnum[sim_id] + ".hdf5")
        except:
            snap_sim = pynbody.load(path1 + "/output/snapshot_" + self.sims_snapnum[sim_id] + ".hdf5")
        assert np.allclose(snap_sim.properties['z'], 0)
        return snap_sim

    def prepare_sim(self, snapshot, sim_id):
        snapshot.physical_units(distance="Mpc h**-1")
        path1 = os.path.dirname(snapshot.filename)
        t0 = time.time()
        try:
            rho = np.load(path1 + "/density_Msol_Mpc-3_h3_z0.npy")
            print("Loaded z=0 density array of simulation ID " + sim_id)
            snapshot["rho"] = rho
            snapshot["rho"].simulation = snapshot
            snapshot["rho"].units = "h**3 Msol Mpc**-3"

        except FileNotFoundError:
            np.save(path1 + "/density_Msol_Mpc-3_h3_z0.npy", snapshot["rho"])
            print("Saved density array of simulation ID " + sim_id)

        rho_m = pynbody.analysis.cosmology.rho_M(snapshot, unit=snapshot["rho"].units)
        den_con = snapshot["rho"] / rho_m
        snapshot.properties['rhoM'] = rho_m
        snapshot["log_den_contrast"] = np.log10(den_con)

        t1 = time.time()
        print("Computing density contrast (z=0) in simulation took " + str((t1 - t0)/60) + " minutes.")

        return snapshot