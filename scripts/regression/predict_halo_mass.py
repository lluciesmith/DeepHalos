import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
import data_processing as dp
import tensorflow as tf
from utils import generator_binary_classification as gbc
from tensorflow.keras.models import load_model


# path_model = "/lfstev/deepskies/luisals/regression/train_sequential_sim0345"
path_model = "/lfstev/deepskies/luisals/regression/train_mixed_sims/lr_decay"
ph = "/lfstev/deepskies/luisals/"
model_file = path_model + "/model/weights.100.hdf5"

# model = load_model(path_model + "/models/model_126_epochs_train_sims0345_3epochs_per_sim.h5")
model = load_model(model_file)

batch_size = 80

h_mass_scaler = dp.get_halo_mass_scaler(["0", "1", "2", "3", "4", "5"])

ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename="random_training_set.txt",
                                                  path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
generator_0 = gbc.create_generator_sim(ids_0, mass_0, batch_size=batch_size,
                                       path=ph + "training_simulation/training_set/")

ids_2, mass_2 = gbc.get_ids_and_regression_labels(sim="2", ids_filename="random_training_set.txt",
                                                  path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
generator_2 = gbc.create_generator_sim(ids_2, mass_2, batch_size=batch_size,
                                       path=ph + "reseed2_simulation/training_set/")

pred_0 = model.predict_generator(generator_0, verbose=1)
halo_mass_pred0 = h_mass_scaler.inverse_transform(pred_0).flatten()
np.save(path_model + "/predictions/pred0_100.npy", halo_mass_pred0)
np.save(path_model + "/predictions/truth0.npy", h_mass_scaler.inverse_transform(mass_0.reshape(-1,1)).flatten())

pred_2 = model.predict_generator(generator_2, verbose=1)
halo_mass_pred2 = h_mass_scaler.inverse_transform(pred_2).flatten()
np.save(path_model + "/predictions/pred2_100.npy", halo_mass_pred2)
np.save(path_model + "/predictions/truth2.npy", h_mass_scaler.inverse_transform(mass_2.reshape(-1,1)).flatten())


# f = plt.figure(figsize=(8, 6))
# plt.plot(np.log10(halo_mass[val_ids]), np.log10(halo_mass[val_ids]), color="dimgrey")
# plt.scatter(np.log10(halo_mass[val_ids]), val_training_log_mass, s=1)
# plt.xlabel("True log mass", fontsize=18)
# plt.ylabel("Predicted log mass", fontsize=18)
# plt.subplots_adjust(bottom=0.14)
# plt.title("Training sim validation set (low-mass haloes)", fontsize=18)
# plt.savefig("training_sim_predictions.pdf")


