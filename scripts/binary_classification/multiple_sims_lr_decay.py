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
import time


if __name__ == "__main__":
    ########### CREATE GENERATORS FOR SIMULATIONS #########

    # ph = "share/hypatia/lls/deep_halos/"
    path_model = "/lfstev/deepskies/luisals/binary_classification/train_mixed_sims/lr_decay/"
    ph = "/lfstev/deepskies/luisals/"

    t0 = time.time()

    with tf.device('/cpu:0'):

        ids_0, labels_0 = gbc.get_ids_and_binary_class_labels(sim="0", threshold=2*10**12)
        ids_3, labels_3 = gbc.get_ids_and_binary_class_labels(sim="3", threshold=2*10**12)
        ids_4, labels_4 = gbc.get_ids_and_binary_class_labels(sim="4", threshold=2*10**12)
        ids_5, labels_5 = gbc.get_ids_and_binary_class_labels(sim="5", threshold=2*10**12)

        sims = ["0", "3", "4", "5"]
        ids_s = [ids_0, ids_3, ids_4, ids_5]
        labels_ids = [labels_0, labels_3, labels_4, labels_5]
        generator_training = gbc.create_generator_multiple_sims(sims, ids_s, labels_ids, batch_size=80)

        ran = np.load("/lfstev/deepskies/luisals/binary_classification/train_mixed_sims/validation_set_indices.npy")
        ids_val = [ids_0[ran], ids_3[ran], ids_4[ran], ids_5[ran]]
        labels_val = [labels_0[ran], labels_3[ran], labels_4[ran], labels_5[ran]]
        generator_val = gbc.create_generator_multiple_sims(sims, ids_val, labels_val, batch_size=80)

        ids_1, labels_1 = gbc.get_ids_and_binary_class_labels(sim="1", threshold=2*10**12)
        generator_1 = gbc.create_generator_sim(ids_1, labels_1, batch_size=80,
                                               path=ph + "reseed1_simulation/reseed1_training/")

    t1 = time.time()
    print("Loading generators took " + str((t1 - t0) / 60) + " minutes to train.")

    ######### TRAINING MODEL ##############

    with tf.device('/gpu:0'):

        # checkpoint
        filepath = path_model + "/weights.{epoch:02d}.hdf5"
        checkpoint_call = callbacks.ModelCheckpoint(filepath, save_freq='epoch')

        # save histories
        csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=False)

        # decay the learning rate
        lr_decay = LearningRateScheduler(CNN.lr_scheduler)

        callbacks_list = [checkpoint_call, csv_logger, lr_decay]

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
                        callbacks=callbacks_list, use_multiprocessing=True, num_epochs=80, workers=8, verbose=1,
                        model_type="binary_classification", lr=0.0001)

        model = Model.model
        history = Model.history

        np.save(path_model + "/history_80_epochs_mixed_sims.npy", history.history)
        model.save(path_model + "/model_80_epochs_mixed_sims.h5")


        # # resume training in case job stops
        # model = load_model(path_model + "weights.09.hdf5")
        #
        # csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
        # callbacks_list = [checkpoint_call, csv_logger, lr_decay]
        #
        # history = model.fit_generator(generator=generator_training, validation_data=generator_1,
        #                               callbacks=callbacks_list, use_multiprocessing=False,
        #                               epochs=81, verbose=1, shuffle=True, workers=1, initial_epoch=9)
        #
        # np.save(path_model + "/history_100_epochs_mixed_sims.npy", history.history)
        # model.save(path_model + "/model_100_epochs_mixed_sims.h5")
