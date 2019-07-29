import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
import CNN
import data_processing as dp
from tensorflow import set_random_seed
from utils import generator_binary_classification as gbc
from tensorflow.keras.models import load_model


if __name__ == "__main__":

    path_model = "/lfstev/deepskies/luisals/regression/train_sequential_sim0345"
    ph = "/lfstev/deepskies/luisals/"

    ########### CREATE GENERATORS FOR SIMULATIONS #########

    batch_size = 80

    h_mass_scaler = dp.get_halo_mass_scaler(["0", "1", "2", "3", "4", "5"])

    ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename="balanced_training_set.txt",
                                                      path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
    generator_0 = gbc.create_generator_sim(ids_0, mass_0, batch_size=batch_size,
                                           path=ph + "training_simulation/training_sim_binary/")

    ids_1, mass_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename="balanced_training_set.txt",
                                                      path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
    generator_1 = gbc.create_generator_sim(ids_1, mass_1, batch_size=batch_size,
                                           path=ph + "reseed1_simulation/reseed1_training/")

    ids_2, mass_2 = gbc.get_ids_and_regression_labels(sim="2", ids_filename="balanced_training_set.txt",
                                                      path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
    generator_2 = gbc.create_generator_sim(ids_2, mass_2, batch_size=batch_size,
                                           path=ph + "reseed2_simulation/reseed2_training/")

    ids_3, mass_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename="balanced_training_set.txt",
                                                      path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
    generator_3 = gbc.create_generator_sim(ids_3, mass_3, batch_size=batch_size,
                                           path=ph + "reseed3_simulation/reseed3_training/")

    ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename="balanced_training_set.txt",
                                                      path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
    generator_4 = gbc.create_generator_sim(ids_4, mass_4, batch_size=batch_size,
                                           path=ph + "reseed4_simulation/reseed4_training/")

    ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename="balanced_training_set.txt",
                                                      path="/lfstev/deepskies/luisals/", fitted_scaler=h_mass_scaler)
    generator_5 = gbc.create_generator_sim(ids_5, mass_5, batch_size=batch_size,
                                           path=ph + "reseed5_simulation/reseed5_training/")

    d = {"0": generator_0, "1": generator_1, "2":generator_2, "3": generator_3, "4": generator_4, "5": generator_5}
    mass_d = {"0": mass_0, "1": mass_1, "2":mass_2, "3": mass_3, "4": mass_4, "5": mass_5}

    ######### TRAINING MODEL ##############

    # set_random_seed(7)
    # param_conv = {'conv_1': {'num_kernels': 5, 'dim_kernel': (3, 3, 3),
    #                          'strides': 2, 'padding': 'valid',
    #                          'pool': True, 'bn': False},
    #               'conv_2': {'num_kernels': 10, 'dim_kernel': (3, 3, 3),
    #                          'strides': 1, 'padding': 'valid',
    #                          'pool': True, 'bn': False},
    #               'conv_3': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
    #                          'strides': 1, 'padding': 'valid',
    #                          'pool': False, 'bn': False}}
    #
    # param_fcc = {'dense_1': {'neurons': 256, 'dropout': 0.2},
    #              'dense_2': {'neurons': 128, 'dropout': 0.2}}
    #
    # Model = CNN.CNN(generator_0, param_conv, param_fcc,
    #                 validation_generator=generator_2, metrics=["mae"], num_gpu=1, use_multiprocessing=True,
    #                 num_epochs=3, workers=8, verbose=1, model_type="regression")
    # model = Model.model
    # history = Model.history
    #
    # h = [history.history]
    #
    # # Run 10 loops of simulations and save at every loop
    #
    # loop = ["0", "3", "4", "5"]
    # sim_list = loop * 10
    #
    # for num, i in enumerate(sim_list[1:], 2):
    #     gen = d[str(i)]
    #     mass_i = mass_d[str(i)]
    #     print("Training on sim " + str(i))
    #
    #     history = model.fit_generator(generator=gen, validation_data=generator_2, use_multiprocessing=False,
    #                                   epochs=3, verbose=1, shuffle=True)
    #     h.append(history.history)
    #
    #     np.save(path_model + "/histories/history_" + str(num*3) + "_epochs_sims_3_per_epoch.npy", h)
    #     model.save(path_model + "/models/model_" + str(num*3) + "_epochs_train_sims0345_3epochs_per_sim.h5")

    ######## RESUME TRAINING ###########

    h = np.load(path_model + "/histories/history_63_epochs_sims_3_per_epoch.npy", allow_pickle=True)
    h = list(h)

    model = load_model(path_model + "/models/model_63_epochs_train_sims0345_3epochs_per_sim.h5")

    loop = ["0", "3", "4", "5"]
    sim_list = loop * 15

    for num, i in enumerate(sim_list[1:], 2):
        if num <= 21:
            pass
        else:
            gen = d[str(i)]
            mass_i = mass_d[str(i)]
            print("Training on sim " + str(i))

            history = model.fit_generator(generator=gen, validation_data=generator_2, use_multiprocessing=False,
                                          epochs=3, verbose=1, shuffle=True)
            h.append(history.history)

            np.save(path_model + "/histories/history_" + str(num * 3) + "_epochs_sims_3_per_epoch.npy", h)
            model.save(path_model + "/models/model_" + str(num * 3) + "_epochs_train_sims0345_3epochs_per_sim.h5")

