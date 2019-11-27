import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
from utils import generators_training as gbc
from tensorflow.keras.models import load_model

if __name__ == "__main__":

    # ph = "share/hypatia/lls/deep_halos/"
    path_model = "/lfstev/deepskies/luisals/regression/z_2.1/"
    ph = "/lfstev/deepskies/luisals/"
    model_file = path_model + "/model_80_epochs_mixed_sims.h5"

    rescale_mean = 240
    rescale_std = 1300


    f = "random_training_set.txt"
    ids_3, mass_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename=f, fitted_scaler=None)
    ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename=f, fitted_scaler=None)
    ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename=f, fitted_scaler=None)
    output_ids, output_scaler = gbc.get_standard_scaler_and_transform([mass_3, mass_4, mass_5])

    ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=output_scaler)
    generator_0 = gbc.create_generator_sim(ids_0, mass_0, batch_size=80,
                                           rescale_mean=rescale_mean, rescale_std=rescale_std, z=2.1,
                                           path=ph + "training_simulation/z2_subboxes/")

    model = load_model(model_file)

    pred_1 = model.predict_generator(generator_0, verbose=1)
    halo_mass_pred1 = output_scaler.inverse_transform(pred_1).flatten()
    np.save(path_model + "/predictions/pred0_80.npy", halo_mass_pred1)
    np.save(path_model + "/predictions/truth0.npy", output_scaler.inverse_transform(mass_0).flatten())

    # f = plt.figure(figsize=(8, 6))
    # plt.plot(np.log10(halo_mass[val_ids]), np.log10(halo_mass[val_ids]), color="dimgrey")
    # plt.scatter(np.log10(halo_mass[val_ids]), val_training_log_mass, s=1)
    # plt.xlabel("True log mass", fontsize=18)
    # plt.ylabel("Predicted log mass", fontsize=18)
    # plt.subplots_adjust(bottom=0.14)
    # plt.title("Training sim validation set (low-mass haloes)", fontsize=18)
    # plt.savefig("training_sim_predictions.pdf")
