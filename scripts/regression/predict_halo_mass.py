import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
from utils import generators_training as gbc
from tensorflow.keras.models import load_model

# 
# batch_size = 80
# 
# h_mass_scaler = dp.get_halo_mass_scaler(["0", "1", "2", "3", "4", "5"])
# 
# ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename="random_training_set.txt",
#                                                   path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
# generator_0 = gbc.create_generator_sim(ids_0, mass_0, batch_size=batch_size,
#                                        path=ph + "training_simulation/training_set/")
# 
# ids_2, mass_2 = gbc.get_ids_and_regression_labels(sim="2", ids_filename="random_training_set.txt",
#                                                   path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
# generator_2 = gbc.create_generator_sim(ids_2, mass_2, batch_size=batch_size,
#                                        path=ph + "reseed2_simulation/training_set/")

########### CREATE GENERATORS FOR SIMULATIONS #########

# ph = "share/hypatia/lls/deep_halos/"
path_model = "/lfstev/deepskies/luisals/regression/train_mixed_sims/standardize_wdropout/"
ph = "/lfstev/deepskies/luisals/"
model_file = path_model + "/model/weights.80.hdf5"

rescale_mean = 1.004
rescale_std = 0.05

model = load_model(model_file)
f = "random_training_set.txt"

ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=None)
ids_3, mass_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename=f, fitted_scaler=None)
ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename=f, fitted_scaler=None)
ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename=f, fitted_scaler=None)
output_ids, output_scaler = gbc.get_standard_scaler_and_transform([mass_0, mass_3, mass_4, mass_5])

# ran_val = np.random.choice(np.arange(20000), 4000)
ids_1, so_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename=f, fitted_scaler=output_scaler)
generator_1 = gbc.create_generator_sim(ids_1, so_1, batch_size=80,
                                       rescale_mean=rescale_mean, rescale_std=rescale_std,
                                       path=ph + "reseed1_simulation/training_set/")
ids_31, so_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename=f, fitted_scaler=output_scaler)
generator_3 = gbc.create_generator_sim(ids_3, so_3, batch_size=80,
                                       rescale_mean=rescale_mean, rescale_std=rescale_std,
                                       path=ph + "reseed3_simulation/training_set/")

np.testing.assert_allclose(ids_3, ids_31)
np.testing.assert_allclose(mass_3, output_scaler.inverse_transform(so_3).flatten())

pred_1 = model.predict_generator(generator_1, verbose=1)
halo_mass_pred1 = output_scaler.inverse_transform(pred_1).flatten()
np.save(path_model + "/predictions/pred1_80.npy", halo_mass_pred1)
np.save(path_model + "/predictions/truth1.npy", output_scaler.inverse_transform(so_1).flatten())

pred_3 = model.predict_generator(generator_3, verbose=1)
halo_mass_pred3 = output_scaler.inverse_transform(pred_3).flatten()
np.save(path_model + "/predictions/pred3_44.npy", halo_mass_pred3)
np.save(path_model + "/predictions/truth3.npy", output_scaler.inverse_transform(so_3).flatten())


# f = plt.figure(figsize=(8, 6))
# plt.plot(np.log10(halo_mass[val_ids]), np.log10(halo_mass[val_ids]), color="dimgrey")
# plt.scatter(np.log10(halo_mass[val_ids]), val_training_log_mass, s=1)
# plt.xlabel("True log mass", fontsize=18)
# plt.ylabel("Predicted log mass", fontsize=18)
# plt.subplots_adjust(bottom=0.14)
# plt.title("Training sim validation set (low-mass haloes)", fontsize=18)
# plt.savefig("training_sim_predictions.pdf")


