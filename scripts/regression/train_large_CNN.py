import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
import tensorflow
from tensorflow.keras.models import load_model
import dlhalos_code.data_processing as tn
import numpy as np
from pickle import dump
import time


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    params_inputs = {'batch_size': 40,
                     'rescale_mean': 1.005,
                     'rescale_std': 0.05050,
                     'dim': (75, 75, 75)
                     }

    # define a common scaler for the output

    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    training_set = tn.InputsPreparation(train_sims, load_ids=True, shuffle=True)
    generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS,
                                          s.sims_dic, **params_inputs)

    validation_set = tn.InputsPreparation([val_sim], load_ids=True, random_subset_all=4000,
                                            scaler_output=training_set.scaler_output, shuffle=True)
    generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS,
                                              s.sims_dic, **params_inputs)
    dump(training_set.scaler_output, open(path_model + 'scaler_output.pkl', 'wb'))

    ######### TRAINING MODEL ##############

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

    # save histories
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)

    callbacks_list = [checkpoint_call, csv_logger]
    tensorflow.compat.v1.set_random_seed(7)

    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  'conv_2': {'num_kernels': 64, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  'conv_3': {'num_kernels': 128, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  'conv_4': {'num_kernels': 256, 'dim_kernel': (3, 3, 3),
                              'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  'conv_5': {'num_kernels': 256, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True}
                  }

    param_fcc = {'dense_1': {'neurons': 1024, 'bn': True, 'dropout': 0.2},
                 'dense_2': {'neurons': 256, 'bn': False, 'dropout': 0.2}}

    Model = CNN.CNN(param_conv, param_fcc, dim=params_inputs['dim'],
                    training_generator=generator_training, validation_generator=generator_validation,
                    validation_freq=1, callbacks=callbacks_list, num_epochs=100,
                    use_multiprocessing=True, workers=2, max_queue_size=10,
                    verbose=1, model_type="regression", lr=0.0001, train=True)

    np.save(path_model + "/history_100_epochs_mixed_sims.npy", Model.history)
    Model.model.save(path_model + "/model_100_epochs_mixed_sims.h5")
