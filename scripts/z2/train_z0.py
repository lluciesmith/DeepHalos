import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from utils import generators_training as gbc
import time
import tensorflow


def take_subset_ids_above_mass_threshold(particle_ids, log_mass_particle_ids, log_mass_threshold=13):
    ind = np.where(log_mass_particle_ids >= log_mass_threshold)[0]
    return particle_ids[ind], log_mass_particle_ids[ind]


if __name__ == "__main__":
    ########### CREATE GENERATORS FOR SIMULATIONS #########

    # ph = "share/hypatia/lls/deep_halos/"
    # path_model = "/lfstev/deepskies/luisals/regression/z0/alpha0.03_5sims/batchnorm"
    # path_model = "/lfstev/deepskies/luisals/regression/z0/highmass"
    path_model = "/lfstev/deepskies/luisals/regression/z0/highmass/51_3"
    ph = "/lfstev/deepskies/luisals/"

    # rescale_mean = 1.239
    # rescale_std = 0.89
    rescale_mean = 0
    rescale_std = 1

    t0 = time.time()

    f = "random_training_set.txt"
    ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=None)
    ids_2, mass_2 = gbc.get_ids_and_regression_labels(sim="2", ids_filename=f, fitted_scaler=None)
    ids_3, mass_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename=f, fitted_scaler=None)
    ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename=f, fitted_scaler=None)
    ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename=f, fitted_scaler=None)

    # restrict ids to those in high-mass halos

    ids_0, mass_0 = take_subset_ids_above_mass_threshold(ids_0, mass_0, log_mass_threshold=13)
    ids_2, mass_2 = take_subset_ids_above_mass_threshold(ids_2, mass_2, log_mass_threshold=13)
    ids_3, mass_3 = take_subset_ids_above_mass_threshold(ids_3, mass_3, log_mass_threshold=13)
    ids_4, mass_4 = take_subset_ids_above_mass_threshold(ids_4, mass_4, log_mass_threshold=13)
    ids_5, mass_5 = take_subset_ids_above_mass_threshold(ids_5, mass_5, log_mass_threshold=13)

    # training set
    sims = ["0", "2", "3", "4", "5"]
    ids_s = [ids_0, ids_2, ids_3, ids_4, ids_5]
    output_ids, output_scaler = gbc.get_standard_scaler_and_transform([mass_0, mass_2, mass_3, mass_4, mass_5])

    generator_training = gbc.create_generator_multiple_sims(sims, ids_s, output_ids, batch_size=100000, dim=(51, 51, 51),
                                                            rescale_mean=rescale_mean, rescale_std=rescale_std, z=0)
    X, y = generator_training[0]

    # validation set
    # ran_val = np.random.choice(np.arange(len(ids_1)), 4000)
    # np.save(path_model + "ran_val1.npy", ran_val)
    ids_1, mass_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename=f, fitted_scaler=output_scaler,
                                                      shuffle=False)
    ids_1, mass_1 = take_subset_ids_above_mass_threshold(ids_1, mass_1, log_mass_threshold=13)

    ran_val = np.random.choice(np.arange(len(ids_1)), 4000)
    generator_1 = gbc.create_generator_sim(ids_1[ran_val], mass_1[ran_val], batch_size=4000, dim=(51, 51, 51),
                                           rescale_mean=rescale_mean, rescale_std=rescale_std, z=0,
                                           path=ph + "reseed1_simulation/z0_subboxes/")
    X_val1, y_val1 = generator_1[0]

    t1 = time.time()
    print("Loading generators took " + str((t1 - t0) / 60) + " minutes.")

    ######### TRAINING MODEL ##############

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, freq='epoch')

    # save histories
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=False)

    # decay the learning rate
    # lr_decay = LearningRateScheduler(CNN.lr_scheduler)

    callbacks_list = [checkpoint_call, csv_logger]
    # callbacks_list = [checkpoint_call, csv_logger, lr_decay]

    tensorflow.compat.v1.set_random_seed(7)

    param_conv = {'conv_1': {'num_kernels': 4, 'dim_kernel': (3, 3, 3),
                             'strides': 2, 'padding': 'valid', 'pool': True, 'bn': False},
                  'conv_2': {'num_kernels': 8, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'valid', 'pool': True, 'bn': False},
                  'conv_3': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'valid',  'pool': True, 'bn': False},
                  # 'conv_4': {'num_kernels': 64, 'dim_kernel': (2, 2, 2),
                  #            'strides': 1, 'padding': 'valid', 'pool': False, 'bn': True}
                  }

    param_fcc = {#'dense_1': {'neurons': 1024, 'bn': False, 'dropout': 0.2},
                 'dense_1': {'neurons': 256, 'bn': False, 'dropout': 0.2},
                 'dense_2': {'neurons': 128, 'bn': False, 'dropout': 0.2}}

    Model = CNN.CNN(generator_training, param_conv, param_fcc,
                    validation_generator=generator_1, validation_freq=1,
                    # metrics=["mae"],
                    callbacks=callbacks_list, use_multiprocessing=True, num_epochs=80,
                    workers=12, verbose=1, model_type="regression", lr=0.0001, train=False)

    history = Model.model.fit(X, y, batch_size=80, verbose=1, epochs=60, validation_data=(X_val1, y_val1),
                              shuffle=True, callbacks=callbacks_list)

    np.save(path_model + "/history_60_epochs_mixed_sims.npy", history.history)
    Model.model.save(path_model + "/model_60_epochs_mixed_sims.h5")

    generator_1 = gbc.create_generator_sim(ids_1, mass_1, batch_size=len(ids_1), dim=(51, 51, 51),
                                           rescale_mean=rescale_mean, rescale_std=rescale_std, z=0,
                                           path=ph + "reseed1_simulation/z0_subboxes/")
    X1, y1 = generator_1[0]

    pred1 = Model.model.predict(X1)
    h_m_pred = output_scaler.inverse_transform(pred1).flatten()
    true1 = output_scaler.inverse_transform(y1).flatten()
    np.save(path_model + "/predicted1_60.npy", h_m_pred)
    np.save(path_model + "/true1_60.npy", true1)
