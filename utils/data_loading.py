import numpy as np
import data_processing as dp
import sklearn.preprocessing
import time
import gc



t00 = time.time()
path = "/lfstev/deepskies/luisals/"
X = {}
for sim in ["0", "2", "4", "5", "6"]:
    t0 = time.time()
    if sim == "0":
        path1 = path + "training_simulation/"
    else:
        path1 = path + "reseed" + sim + "_simulation/"
    X[sim] = np.load(path1 + "res_75_20000_subboxes.npy")
    t1 = time.time()
    print("Loading one sim took " + str((t1 - t0) / 60) + " minutes.")
    gc.collect()
t2 = time.time()
print("Loading whole training set took " + str((t2 - t00) / 60) + " minutes.")


class DataProcessing:
    def __init__(self, train_sims, test_sims, ids_filename="random_training_set.txt",
                 path="/lfstev/deepskies/luisals/"):

        self.train_sims = train_sims
        self.test_sims = test_sims
        self.ids_filename = ids_filename
        self.path = path

        self.training_ids = [self.get_ids_simulation(sim) for sim in self.train_sims]
        self.testing_ids = [self.get_ids_simulation(sim) for sim in self.test_sims]

        self.X_train, self.input_scaler = self.inputs_training_set()
        self.y_train, self.output_scaler = self.output_training_set()

        self.X_test = self.inputs_test_set()
        self.y_test = self.output_test_set()

    def inputs_training_set(self):
        X = [self.get_features(sim) for sim in self.train_sims]
        rescaled_inputs, input_scaler = self.fit_standard_scaler_and_transform(X, use_subset=True)
        return rescaled_inputs, input_scaler

    def inputs_test_set(self):
        X = [self.get_features(sim, fitted_scaler=self.input_scaler) for sim in self.train_sims]
        return X

    def output_training_set(self):
        mass_ids = [self.get_regression_labels(sim=sim, fitted_scaler=None) for sim in self.train_sims]
        rescaled_outputs, output_scalar = self.fit_standard_scaler_and_transform(mass_ids)
        return np.concatenate(rescaled_outputs), output_scalar

    def output_test_set(self):
        mass_ids = [self.get_regression_labels(sim=sim, fitted_scaler=self.output_scaler) for sim in self.test_sims]
        return mass_ids

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
        Xi_T = np.transpose(Xi, axes=(0, 2, 1, 3))

        if fitted_scaler is not None:
            inputs_rescaled = self.transform_array_given_scaler(fitted_scaler, Xi_T)
        else:
            inputs_rescaled = Xi_T

        return inputs_rescaled

    def fit_standard_scaler_and_transform(self, list_outputs, use_subset=True):
        if use_subset is True:
            ind = np.random.choice(range(list_outputs.dim[1]), 1000, replace=False)
            Xsubset_for_rescaling = [Xi[ind] for Xi in X]
        else:
            Xsubset_for_rescaling = list_outputs
        outputs_conc = np.concatenate(Xsubset_for_rescaling)

        norm_scaler = sklearn.preprocessing.StandardScaler()
        norm_scaler.fit(outputs_conc.reshape(-1, 1))

        rescaled_out = [self.transform_array_given_scaler(norm_scaler, out_i) for out_i in list_outputs]
        return rescaled_out, norm_scaler

    def transform_array_given_scaler(self, scaler, array):
        scaled_array = scaler.transform(array.reshape(-1, 1)).flatten()
        return scaled_array
