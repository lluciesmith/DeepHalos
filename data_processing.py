import numpy as np
from tensorflow.keras.utils import Sequence
import sklearn.preprocessing


class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels, batch_size=40, dim=(51, 51, 51), n_channels=1, shuffle=False,
                 saving_path="/share/data2/lls/deep_halos/subboxes/subbox_51_particle_",
                 halo_masses="", multiple_sims=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.path = saving_path
        self.halo_masses = halo_masses
        self.multiple_sims = multiple_sims
        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate a batch of data """
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        if self.multiple_sims is True:
            X, y = self.__data_generation_multiple_sims(list_IDs_temp)
        else:
            X, y = self.__data_generation(list_IDs_temp)
        return X, y

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
            if self.path == "truth":
                s = np.ones((51, 51, 51)) * self.labels[ID]
            # elif self.path == "sphere":
            #     s = sb.get_sphere_in_box(self.halo_masses[ID])
            else:
                s = np.load(self.path + ID + '/subbox_51_particle_' + ID + '.npy')

            if self.dim != (51, 51, 51):
                low_idx = int(25 - (self.dim[0] - 1)/2)
                high_idx = int(25 + (self.dim[0] - 1)/2 + 1)
                s = s[low_idx:high_idx, low_idx:high_idx, low_idx:high_idx]

            # Take the transpose in order for these to match pynbody's output
            s_t = np.transpose(s, axes=(1, 0, 2))

            X[i] = s_t.reshape((*self.dim, self.n_channels))
            y[i] = self.labels[ID]

        return X, y

    def __data_generation_multiple_sims(self, list_IDs_temp):
            """ Loads data containing batch_size samples """

            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            y = np.empty((self.batch_size,))

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                sim_index = ID[4]
                particle_ID = ID[9:]
                if sim_index == "0":
                    s = np.load(self.path + 'training_simulation/training_set/' + particle_ID +
                                '/subbox_51_particle_' + particle_ID + '.npy')
                else:
                    s = np.load(self.path + "reseed" + sim_index + "_simulation/training_set/" +
                                particle_ID + '/subbox_51_particle_' + particle_ID + '.npy')

                if self.dim != (51, 51, 51):
                    low_idx = int(25 - (self.dim[0] - 1) / 2)
                    high_idx = int(25 + (self.dim[0] - 1) / 2 + 1)
                    s = s[low_idx:high_idx, low_idx:high_idx, low_idx:high_idx]

                # Take the transpose in order for these to match pynbody's output
                s_t = np.transpose(s, axes=(1, 0, 2))

                X[i] = s_t.reshape((*self.dim, self.n_channels))
                y[i] = self.labels[ID]

            return X, y


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

