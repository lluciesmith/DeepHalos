import numpy as np
import sklearn.preprocessing
import time
import gc
from pickle import load, dump


class DataProcessing:
    def __init__(self, train_sims, test_sims, ids_filename="random_training_set.txt",
                 load_inputs_standard_scaler=False, save_inputs_standard_scaler=False, path_inputs_standard_scaler=".",
                 path="/lfstev/deepskies/luisals/"):

        self.train_sims = train_sims
        self.test_sims = test_sims
        self.ids_filename = ids_filename
        self.path = path

        self.load_inputs_standard_scaler = load_inputs_standard_scaler
        self.save_inputs_standard_scaler = save_inputs_standard_scaler
        self.path_inputs_standard_scaler = path_inputs_standard_scaler

        self.training_ids = [self.get_ids_simulation(sim) for sim in self.train_sims]
        self.testing_ids = [self.get_ids_simulation(sim) for sim in self.test_sims]

        t0 = time.time()
        self.X_train, (self.X_mean, self.X_std) = self.inputs_training_set()
        t1 = time.time()
        print("Loading features training set + rescaling took " + str((t1 - t0) / 60) + " minutes.")

        t0 = time.time()
        self.y_train, self.output_scaler = self.output_training_set()
        t1 = time.time()
        print("Loading output training set + rescaling took " + str((t1 - t0) / 60) + " minutes.")

        t0 = time.time()
        self.X_test = self.inputs_test_set()
        t1 = time.time()
        print("Loading features test set + rescaling took " + str((t1 - t0) / 60) + " minutes.")

        t0 = time.time()
        self.y_test = self.output_test_set()
        t1 = time.time()
        print("Loading output test set + rescaling took " + str((t1 - t0) / 60) + " minutes.")

    def inputs_training_set(self):
        X = [self.get_features(sim) for sim in self.train_sims]
        X_mean, X_std = self.get_mean_std_array(X, use_subset=True)
        rescaled_inputs = self.rescale_features(X, X_mean, X_std)
        return np.concatenate(rescaled_inputs, axis=0), (X_mean, X_std)

    def inputs_test_set(self):
        X = [self.get_features(sim, fitted_scaler=None) for sim in self.train_sims]
        X_rescaled = [(Xi - self.X_mean)/self.X_std for Xi in X]
        return np.concatenate(X_rescaled, axis=0)

    def output_training_set(self):
        mass_ids = [self.get_regression_labels(sim=sim, fitted_scaler=None) for sim in self.train_sims]
        rescaled_outputs, output_scalar = self.fit_standard_scaler_and_transform(mass_ids)
        return np.concatenate(rescaled_outputs), output_scalar

    def output_test_set(self):
        mass_ids = [self.get_regression_labels(sim=sim, fitted_scaler=self.output_scaler) for sim in self.test_sims]
        return np.concatenate(mass_ids)

    def get_ids_simulation(self, sim):
        if sim == "0":
            path1 = self.path + "training_simulation/training_sim_"
        else:
            path1 = self.path + "reseed" + sim + "_simulation/reseed_" + sim + "_"

        with open(path1 + self.ids_filename, "r") as f:
            ids_bc = np.array([line.rstrip("\n") for line in f]).astype("int")

        return ids_bc

    def get_regression_labels(self, sim="0", fitted_scaler=None):
        if sim == "0":
            halo_mass = np.load(self.path + "training_simulation/halo_mass_particles.npy")
        else:
            halo_mass = np.load(self.path + "reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")

        ids = self.get_ids_simulation(sim)

        if fitted_scaler is not None:
            output_ids = self.transform_array_given_scaler(fitted_scaler, np.log10(halo_mass[ids]))
        else:
            output_ids = np.log10(halo_mass[ids])

        return output_ids

    def get_features(self, sim="0", fitted_scaler=None):
        if sim == "0":
            path1 = self.path + "training_simulation/"
        else:
            path1 = self.path + "reseed" + sim + "_simulation/"

        Xi = np.load(path1 + "res_75_20000_subboxes.npy")
        Xi_T = np.transpose(Xi, axes=(0, 2, 1, 3, 4))

        if fitted_scaler is not None:
            inputs_rescaled = self.transform_array_given_scaler(fitted_scaler, Xi_T)
        else:
            inputs_rescaled = Xi_T
        print("Done features sim " + sim)
        return inputs_rescaled

    def get_mean_std_array(self, list_arrays, use_subset=True):
        if use_subset is True:
            ind = np.random.choice(range(len(list_arrays[0])), 1000, replace=False)
            Xsubset_for_rescaling = [Xi[ind] for Xi in list_arrays]
        else:
            Xsubset_for_rescaling = list_arrays
        outputs_conc = np.concatenate(Xsubset_for_rescaling)
        print("Done concatenating")

        return np.mean(outputs_conc), np.std(outputs_conc)

    def rescale_features(self, list_arrays, mean, std):
        array_rescaled = [(x_i - mean)/std for x_i in list_arrays]
        return array_rescaled

    def fit_standard_scaler_and_transform(self, list_outputs, use_subset=True):
        if use_subset is True:
            ind = np.random.choice(range(len(list_outputs[0])), 1000, replace=False)
            Xsubset_for_rescaling = [Xi[ind] for Xi in list_outputs]
        else:
            Xsubset_for_rescaling = list_outputs
        outputs_conc = np.concatenate(Xsubset_for_rescaling)
        print("Done concatenating")

        if self.load_inputs_standard_scaler is True:
            print("Load the standard scaler")
            norm_scaler = load(open(self.path_inputs_standard_scaler + 'inputs_standard_scaler.pkl', 'rb'))

        else:
            print("Fit the standard scaler")
            norm_scaler = sklearn.preprocessing.StandardScaler()
            norm_scaler.fit(outputs_conc.reshape(-1, 1))
            print("Done fitting the standard scaler")

            if self.save_inputs_standard_scaler is True:
                dump(norm_scaler, open(self.path_inputs_standard_scaler + 'inputs_standard_scaler.pkl', 'wb'))

        rescaled_out = [self.transform_array_given_scaler(norm_scaler, out_i) for out_i in list_outputs]
        print("Done transforming")
        return rescaled_out, norm_scaler

    def transform_array_given_scaler(self, scaler, array):
        scaled_array = scaler.transform(array.reshape(-1, 1)).flatten()
        return scaled_array



# def get_features(sim="0"):
#     if sim == "0":
#         path1 = path + "training_simulation/"
#     else:
#         path1 = path + "reseed" + sim + "_simulation/"
#     Xi = np.load(path1 + "res_75_20000_subboxes.npy")
#     Xi_T = np.transpose(Xi, axes=(0, 2, 1, 3, 4))
#     print("Done features sim " + sim)
#     return Xi_T