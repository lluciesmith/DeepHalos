import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow import set_random_seed
from utils import generator_binary_classification as gbc


if __name__ == "__main__":
    ########### CREATE GENERATORS FOR SIMULATIONS #########

    # ph = "share/hypatia/lls/deep_halos/"
    path_model = "/lfstev/deepskies/luisals/binary_classification/train_mixed_sims"
    ph = "/lfstev/deepskies/luisals/"

    ids_0, labels_0 = gbc.get_ids_and_binary_class_labels(sim="0", threshold=2*10**12)
    ids_3, labels_3 = gbc.get_ids_and_binary_class_labels(sim="3", threshold=2*10**12)
    ids_4, labels_4 = gbc.get_ids_and_binary_class_labels(sim="4", threshold=2*10**12)
    ids_5, labels_5 = gbc.get_ids_and_binary_class_labels(sim="5", threshold=2*10**12)

    sims = ["0", "3", "4", "5"]
    ids_s = [ids_0, ids_3, ids_4, ids_5]
    labels_ids = [labels_0, labels_3, labels_4, labels_5]
    generator_training = gbc.create_generator_multiple_sims(sims, ids_s, labels_ids, batch_size=80)

    ids_1, labels_1 = gbc.get_ids_and_binary_class_labels(sim="1", threshold=2*10**12)
    generator_1 = gbc.create_generator_sim(ids_1, labels_1, path=ph + "reseed1_simulation/reseed1_training/")

    ######### TRAINING MODEL ##############

    # auc
    callback_data = (generator_1, labels_1)
    auc_call = CNN.AucCallback((generator_training, np.concatenate(labels_ids)),
                               callback_data, name_training="i", names_val="1")

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, save_freq='epoch')

    # save histories

    csv_logger = CSVLogger(path_model + "/model/training.log", separator=',', append=False)

    callbacks_list = [auc_call, checkpoint_call, csv_logger]

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
                    callbacks=callbacks_list, use_multiprocessing=True, num_epochs=100, workers=80, verbose=1,
                    model_type="binary_classification")
    model = Model.model
    history = Model.history

    np.save(path_model + "/history_100_epochs_mixed_sims.npy", history.history)
    model.save(path_model + "/model_100_epochs_mixed_sims.h5")


