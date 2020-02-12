import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
from utils import generators_training as gbc
from tensorflow.keras.models import load_model

if __name__ == "__main__":

    # ph = "share/hypatia/lls/deep_halos/"
    path_model = "/lfstev/deepskies/luisals/regression/z_2.1_500/"
    ph = "/lfstev/deepskies/luisals/"
    model_file = path_model + "/model/weights.03.hdf5"

    model = load_model(model_file)

    rescale_mean = 0
    rescale_std = 1

    f = "random_training_set.txt"
    ids_2, mass_2 = gbc.get_ids_and_regression_labels(sim="2", ids_filename=f, fitted_scaler=None)
    ids_3, mass_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename=f, fitted_scaler=None)
    ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename=f, fitted_scaler=None)
    ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename=f, fitted_scaler=None)
    output_ids, output_scaler = gbc.get_standard_scaler_and_transform([mass_2, mass_3, mass_4, mass_5])

    ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=output_scaler)
    generator_0 = gbc.create_generator_sim(ids_0, mass_0, batch_size=80,
                                           rescale_mean=rescale_mean, rescale_std=rescale_std, z=2.1,
                                           path=ph + "training_simulation/z2_subboxes/")

    pred_0 = model.predict_generator(generator_0, verbose=1)
    halo_mass_pred0 = output_scaler.inverse_transform(pred_0).flatten()
    np.save(path_model + "/predictions/pred0_80.npy", halo_mass_pred0)
    np.save(path_model + "/predictions/truth0.npy", output_scaler.inverse_transform(mass_0).flatten())

    ids_1, mass_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename=f, fitted_scaler=output_scaler)
    generator_1 = gbc.create_generator_sim(ids_1, mass_1, batch_size=80,
                                           rescale_mean=rescale_mean, rescale_std=rescale_std, z=2.1,
                                           path=ph + "reseed1_simulation/z2_subboxes/")

    pred_1 = model.predict_generator(generator_1, verbose=1)
    halo_mass_pred1 = output_scaler.inverse_transform(pred_1).flatten()
    np.save(path_model + "/predictions/pred1_80.npy", halo_mass_pred1)
    np.save(path_model + "/predictions/truth1.npy", output_scaler.inverse_transform(mass_1).flatten())

    # f = plt.figure(figsize=(8, 6))
    # plt.plot(np.log10(halo_mass[val_ids]), np.log10(halo_mass[val_ids]), color="dimgrey")
    # plt.scatter(np.log10(halo_mass[val_ids]), val_training_log_mass, s=1)
    # plt.xlabel("True log mass", fontsize=18)
    # plt.ylabel("Predicted log mass", fontsize=18)
    # plt.subplots_adjust(bottom=0.14)
    # plt.title("Training sim validation set (low-mass haloes)", fontsize=18)
    # plt.savefig("training_sim_predictions.pdf")