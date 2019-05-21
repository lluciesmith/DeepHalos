import numpy as np
from tensorflow.keras.utils import Sequence
import CNN
from tensorflow import set_random_seed


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


if __name__ == "__main__":
    p_ids = np.load("/Users/lls/Documents/deep_halos_files/particles.npy")
    halo_mass = np.load("/Users/lls/Documents/mlhalos_files/halo_mass_particles.npy")
    p_inputs = np.load("/Users/lls/Documents/deep_halos_files/3d_inputs_particles.npy")

    # training_ids = np.random.choice(p_ids, 328, replace=False)
    # testing_ids = p_ids[~np.in1d(p_ids, training_ids)]
    training_ids = p_ids
    batch_s = 60

    params = {'dim': (17, 17, 17),
              'batch_size': batch_s,
              'n_channels': 1,
              'shuffle': True}

    training_generator = DataGenerator(training_ids, np.log10(halo_mass), **params)
    # validation_generator = DataGenerator(testing_ids, np.log10(halo_mass), **params)

    set_random_seed(7)

    num_epochs = 5
    num_conv = 2
    num_kernels = [10, 15]
    dim_kernel = [(7, 7, 7), (3, 3, 3)]
    strides = [2, 2]

    Model = CNN.model_w_layers(input_shape_box=(17, 17, 17, 1), num_convolutions=num_conv, num_kernels=num_kernels,
                               dim_kernel=dim_kernel, strides=strides, padding='valid', data_format="channels_last",
                               alpha_relu=0.3, pool=False)

    Model.fit_generator(generator=training_generator,
                        # validation_data=validation_generator,
                        # use_multiprocessing=True,
                        # workers=2)
                        verbose=1, epochs=num_epochs)

    pred_cnn = Model.predict_generator(training_generator)
