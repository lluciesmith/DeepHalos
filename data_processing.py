import numpy as np
from tensorflow.keras.utils import Sequence
import CNN
from tensorflow import set_random_seed
import sklearn.preprocessing


class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels, batch_size=40, dim=(51, 51, 51), n_channels=1, shuffle=True,
                 saving_path="/share/data2/lls/deep_halos/subboxes"):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path = saving_path
        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate a batch of data """
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """ Loads data containing batch_size samples """
        inputs = np.load("/Users/lls/Documents/deep_halos_files/3d_inputs_particles.npy")
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, ))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # X[i,] = np.load(self.path + '/subbox_51_particle_' + ID + '.npy"')
            X[i,] = inputs[i].reshape(17, 17, 17, 1)
            y[i] = self.labels[ID]

        return X, y


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

