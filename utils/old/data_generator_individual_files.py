import numpy as np
from tensorflow.keras.utils import Sequence
import sklearn.preprocessing


class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels, batch_size=40, dim=(51, 51, 51), n_channels=1, shuffle=False,
                 saving_path="/share/data2/lls/deep_halos/subboxes/subbox_51_particle_",
                 halo_masses="", multiple_sims=False, rescale_mean=0, rescale_std=1, z=99):
        self.dim = dim
        self.res = dim[0]
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle

        self.n_channels = n_channels
        self.path = saving_path
        self.halo_masses = halo_masses

        self.multiple_sims = multiple_sims
        self.rescale_mean = rescale_mean
        self.rescale_std = rescale_std

        self.z = z
        if z == 99:
            self._transpose_input = True
        else:
            self._transpose_input = False

        # if self.rescale_input is True:
        #     mass_variance = np.load("/lfstev/deepskies/luisals/mass_variance_smoothing_scales.npy")
        #     self.v = mass_variance[self.index_scale]

        self.on_epoch_end()

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate a batch of data """
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        if self.multiple_sims is True:
            print("Starting to load data from multiple sims..")
            X, y = self.__data_generation_multiple_sims(list_IDs_temp)

        else:
            print("Starting to load data from one sim..")
            X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            print("Be very aware of setting shuffle=True! I think there is a bug here")
            np.random.shuffle(self.indexes)

    def _process_input(self, matrix):

        if self.dim < (51, 51, 51):
            low_idx = int(25 - (self.dim[0] - 1) / 2)
            high_idx = int(25 + (self.dim[0] - 1) / 2 + 1)
            s = matrix[low_idx:high_idx, low_idx:high_idx, low_idx:high_idx]
        else:
            s = matrix

        # Take the transpose in order for these to match pynbody's output
        if self._transpose_input is True:
            s_t = np.transpose(s, axes=(1, 0, 2))
        else:
            s_t = np.log10(1 + s)

        # Rescale inputs
        s_t_rescaled = (s_t - self.rescale_mean) / self.rescale_std

        return s_t_rescaled.reshape((*self.dim, self.n_channels))

    def __data_generation(self, list_IDs_temp):
        """ Loads data containing batch_size samples """

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            s = np.load(self.path + ID + '/subbox_' + self.res + '_particle_' + ID + '.npy')

            X[i] = self._process_input(s)
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

                if self.z == 99:
                    if self.res == 51:
                        path_midddle = "training_set/"
                    elif self.res == 75:
                        path_midddle = "training_set_res75/"
                    else:
                        raise(ValueError, "I have subboxes only for 51 or 75 cubed resolution.")
                elif self.z == 2.1:
                    path_midddle = "z2_subboxes_500/"
                elif self.z == 0.5:
                    path_midddle = "z05_subboxes/"
                elif self.z == 0:
                    path_midddle = "z0_subboxes/"
                else:
                    raise(ValueError, "Get subboxes from z=99, z=2 or z=0.")

                if sim_index == "0":
                    s = np.load(self.path + 'training_simulation/' + path_midddle + particle_ID +
                                '/subbox_' + str(self.res) + '_particle_' + particle_ID + '.npy')
                else:
                    s = np.load(self.path + "reseed" + sim_index + "_simulation/" + path_midddle +
                                particle_ID + '/subbox_' + str(self.res) + '_particle_' + particle_ID + '.npy')

                X[i] = self._process_input(s)
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


def normalise_distribution_to_given_variance(samples, variance,
                                             mean_samples=1.0040703121496461, std_samples=0.05050898332331686):
    samples_transformed = ((samples - mean_samples)/std_samples * np.sqrt(variance)) + mean_samples
    return samples_transformed

# class TfData():
#     def __init(self):
#         self.FLAGS = {}
#
#
#
#     def parse_fn(self, example):
#         "Parse TFExample records and perform simple data augmentation."
#         example_fmt = {
#             "image": tf.FixedLengthFeature((), tf.string, ""),
#             "label": tf.FixedLengthFeature((), tf.int64, -1)
#         }
#         parsed = tf.parse_single_example(example, example_fmt)
#         image = tf.image.decode_image(parsed["image"])
#         image = _augment_helper(image)  # augments image using slice, reshape, resize_bilinear
#         return image, parsed["label"]
#
#     def input_fn(self, FLAGS):
#         files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
#         dataset = files.interleave(tf.data.TFRecordDataset)
#         dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
#         dataset = dataset.apply(tf.contrib.data.map_and_batch(
#             map_func=parse_fn, batch_size=FLAGS.batch_size))
#         dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
#         return dataset

