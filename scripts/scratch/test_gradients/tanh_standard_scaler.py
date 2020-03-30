import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import dlhalos_code.data_processing as tn
import numpy as np
from pickle import dump
import time
from tensorflow.keras.callbacks import TensorBoard


# if __name__ == "__main__":

import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import dlhalos_code.data_processing as tn
import numpy as np
from pickle import dump
import time
from tensorflow.keras.callbacks import TensorBoard


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/tanh2/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    params_inputs = {'batch_size': 40,
                     'rescale_mean': 1.005,
                     'rescale_std': 0.05050,
                     'dim': (75, 75, 75)
                     }
    params_inputs_val = {'batch_size': 20,
                     'rescale_mean': 1.005,
                     'rescale_std': 0.05050,
                     'dim': (75, 75, 75)
                     }

        # define a common scaler for the output

    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    training_set = tn.InputsPreparation(train_sims, load_ids=False, random_subset_each_sim=5000, shuffle=True)
    generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS,
                                          s.sims_dic, **params_inputs)

    validation_set = tn.InputsPreparation([val_sim], load_ids=True, random_subset_all=4000,
                                            scaler_output=training_set.scaler_output, shuffle=True)
    generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS,
                                              s.sims_dic, **params_inputs_val)
    dump(training_set.scaler_output, open(path_model + 'scaler_output.pkl', 'wb'))

    ######### TRAINING MODEL ##############

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

    # save histories
    csv_logger = CSVLogger(path_model + "/training.log", separator=',')

    # tensorboard
    tb = TensorBoard(log_dir=path_model + '/logs', histogram_freq=5, update_freq='epoch',
                     write_grads=True, write_graph=False)

    callbacks_list = [checkpoint_call, csv_logger, tb]
    # callbacks_list = [checkpoint_call, csv_logger]
    tensorflow.compat.v1.set_random_seed(7)

    kernel_reg = regularizers.l2(0.0005)
    bias_reg = regularizers.l2(0.0005)
    activation = 'tanh'
    relu = False

    param_conv = {'conv_1': {'num_kernels': 16, 'dim_kernel': (3, 3, 3), 'activation': activation, 'relu': relu,
                             'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg,
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': False},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'activation': activation, 'relu': relu,
                             'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg,
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': False},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'activation': activation, 'relu': relu,
                             'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg,
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': False},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'activation': activation, 'relu': relu,
                             'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg,
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': False},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'activation': activation, 'relu': relu,
                             'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg,
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': False}
                  }

    param_fcc = {'dense_1': {'neurons': 256, 'bn': False, 'dropout': 0.4, 'activation': activation, 'relu': relu,
                             'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg},
                 'dense_2': {'neurons': 128, 'bn': False, 'dropout': 0.4, 'activation': activation, 'relu': relu,
                             'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg},
                 'last': {'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg}
                 }

    Model = CNN.CNN(param_conv, param_fcc, model_type="regression",
                    training_generator=generator_training, validation_generator=generator_validation,
                    lr=0.001, callbacks=callbacks_list, metrics=['mae', 'mse'],
                    num_epochs=100, dim=params_inputs['dim'],
                    max_queue_size=10, use_multiprocessing=True, workers=2, verbose=1,
                    num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1, train=True)

    np.save(path_model + "/history_100_epochs_mixed_sims.npy", Model.history)
    Model.model.save(path_model + "/model_100_epochs_mixed_sims.h5")

