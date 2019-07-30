import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
import tensorflow as tf
import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import set_random_seed
from utils import generator_binary_classification as gbc
from tensorflow.keras.models import load_model
import data_processing as dp
import time


if __name__ == "__main__":
    ########### CREATE GENERATORS FOR SIMULATIONS #########

    # ph = "share/hypatia/lls/deep_halos/"
    path_model = "/lfstev/deepskies/luisals/regression/train_mixed_sims/"
    ph = "/lfstev/deepskies/luisals/"

    t0 = time.time()

    with tf.device('/cpu:0'):
        h_mass_scaler = dp.get_halo_mass_scaler(["0", "1", "2", "3", "4", "5"])
        f = "random_training_set.txt"

        ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=h_mass_scaler)
        ids_3, mass_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename=f, fitted_scaler=h_mass_scaler)
        ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename=f, fitted_scaler=h_mass_scaler)
        ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename=f, fitted_scaler=h_mass_scaler)

        sims = ["0", "3", "4", "5"]
        ids_s = [ids_0, ids_3, ids_4, ids_5]
        mass_ids = [mass_0, mass_3, mass_4, mass_5]
        generator_training = gbc.create_generator_multiple_sims(sims, ids_s, mass_ids, batch_size=80)

        ran = np.random.choice(np.arange(len(ids_0)), 5000, replace=False)
        np.save(path_model + "/validation_set_indices.npy", ran)
        ids_val = [ids_0[ran], ids_3[ran], ids_4[ran], ids_5[ran]]
        mass_val = [mass_0[ran], mass_3[ran], mass_4[ran], mass_5[ran]]
        generator_val = gbc.create_generator_multiple_sims(sims, ids_val, mass_val, batch_size=80)

        ids_1, mass_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename=f, fitted_scaler=h_mass_scaler)
        generator_1 = gbc.create_generator_sim(ids_1, mass_1, batch_size=80,
                                               path=ph + "reseed1_simulation/training_set/")

    t1 = time.time()
    print("Loading generators took " + str((t1 - t0) / 60) + " minutes to train.")

    ######### TRAINING MODEL ##############

    with tf.device('/gpu:0'):

        # checkpoint
        filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
        checkpoint_call = callbacks.ModelCheckpoint(filepath, save_freq='epoch')

        # save histories
        csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=False)

        callbacks_list = [checkpoint_call, csv_logger]

        set_random_seed(7)
        param_conv = {'conv_1': {'num_kernels': 5, 'dim_kernel': (3, 3, 3),
                                 'strides': 2, 'padding': 'valid',
                                 'pool': True, 'bn': False},
                      'conv_2': {'num_kernels': 10, 'dim_kernel': (3, 3, 3),
                                 'strides': 1, 'padding': 'valid',
                                 'pool': True, 'bn': False},
                      'conv_3': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
                                 'strides': 1, 'padding': 'valid',
                                 'pool': False, 'bn': False},
                      }

        param_fcc = {'dense_1': {'neurons': 256, 'dropout': 0.2},
                     'dense_2': {'neurons': 128, 'dropout': 0.2}}

        Model = CNN.CNN(generator_training, param_conv, param_fcc, validation_generator=generator_1,
                        callbacks=callbacks_list, use_multiprocessing=True, num_epochs=100, workers=14, verbose=1,
                        model_type="regression", lr=0.0001)

        model = Model.model
        history = Model.history

        np.save(path_model + "/history_100_epochs_mixed_sims.npy", history.history)
        model.save(path_model + "/model_100_epochs_mixed_sims.h5")

