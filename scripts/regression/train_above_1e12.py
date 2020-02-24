import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from utils import generators_training_old as gbc
import time


def take_subset_ids_above_mass_threshold(particle_ids, log_mass_particle_ids, log_mass_threshold=13):
    ind = np.where(log_mass_particle_ids >= log_mass_threshold)[0]
    return particle_ids[ind], log_mass_particle_ids[ind]

if __name__ == "__main__":
    t0 = time.time()

    ########### CREATE GENERATORS FOR SIMULATIONS #########

    path_model = "/lfstev/deepskies/luisals/regression/train_mixed_sims/51_3_above_1e12/"
    ph = "/lfstev/deepskies/luisals/"

    f = "random_training_set.txt"
    sims = ["0", "2", "3", "4", "5"]

    ids_s = []
    m_ids = []
    for sim in sims:
        ids_i, mass_i = gbc.get_ids_and_regression_labels(sim=sim, ids_filename=f, fitted_scaler=None, shuffle=False)
        ids_i, mass_i = take_subset_ids_above_mass_threshold(ids_i, mass_i, log_mass_threshold=12)
        ids_s.append(ids_i)
        m_ids.append(mass_i)

    output_ids, output_scaler = gbc.get_standard_scaler_and_transform(m_ids)

    # training set

    generator_training = gbc.create_generator_multiple_sims(sims, ids_s, output_ids,
                                                            # batch_size=10,
                                                            batch_size=len(np.concatenate(output_ids)),
                                                            dim=(51, 51, 51),
                                                            rescale_mean=0, rescale_std=1, z=99)
    X, y = generator_training[0]
    X_mean = X.mean()
    X_std = X.std()
    X_rescaled = (X - X_mean)/X_std

    # validation set
    ids_1, mass_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename=f, fitted_scaler=None, shuffle=False)
    ids_1, mass_1 = take_subset_ids_above_mass_threshold(ids_1, mass_1, log_mass_threshold=12)
    mass_1 = gbc.transform_array_given_scaler(output_scaler, mass_1)

    ran_val = np.random.choice(np.arange(len(ids_1)), 4000)
    generator_1 = gbc.create_generator_sim(ids_1[ran_val], mass_1[ran_val],
                                           batch_size=len(ids_1),
                                           # batch_size=10,
                                           dim=(51, 51, 51),
                                           rescale_mean=X_mean, rescale_std=X_std, z=99,
                                           path=ph + "reseed1_simulation/training_set/")
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
    lr_decay = LearningRateScheduler(CNN.lr_scheduler)
    # callbacks_list = [checkpoint_call, csv_logger]
    callbacks_list = [checkpoint_call, csv_logger, lr_decay]

    #tensorflow.compat.v1.set_random_seed(7)

    param_conv = {'conv_1': {'num_kernels': 4, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': False},
                  'conv_2': {'num_kernels': 8, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': False},
                  'conv_3': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same',  'pool': "max", 'bn': False},
                  'conv_4': {'num_kernels': 4, 'dim_kernel': (1, 1, 1),
                              'strides': 1, 'padding': 'same', 'pool': None, 'bn': True}
                  }

    param_fcc = {#'dense_1': {'neurons': 1024, 'bn': False, 'dropout': 0.2},
                 'dense_1': {'neurons': 256, 'bn': False, 'dropout': 0.4},
                 'dense_2': {'neurons': 128, 'bn': False, 'dropout': 0.4}}

    Model = CNN.CNN(param_conv, param_fcc, training_generator=generator_training,
                    validation_generator=generator_1, validation_freq=1,
                    callbacks=callbacks_list, use_multiprocessing=True, num_epochs=80,
                    workers=12, verbose=1, model_type="regression", lr=0.0001, train=False)

    history = Model.model.fit(X_rescaled, y, batch_size=80, verbose=1, epochs=100, validation_data=(X_val1, y_val1),
                              shuffle=True, callbacks=callbacks_list)

    np.save(path_model + "/history_60_epochs_mixed_sims.npy", history.history)
    Model.model.save(path_model + "/model_60_epochs_mixed_sims.h5")


    # test data

    generator_1 = gbc.create_generator_sim(ids_1, mass_1, batch_size=len(ids_1), dim=(51, 51, 51),
                                           rescale_mean=X_mean, rescale_std=X_std, z=99,
                                           path=ph + "reseed1_simulation/training_set/")
    X1, y1 = generator_1[0]

    pred1 = Model.model.predict(X1)
    h_m_pred = output_scaler.inverse_transform(pred1).flatten()
    true1 = output_scaler.inverse_transform(y1).flatten()
    np.save(path_model + "/predicted1_60.npy", h_m_pred)
    np.save(path_model + "/true1_60.npy", true1)
