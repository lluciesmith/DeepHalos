import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from dlhalos_code import evaluation as evalu
from pickle import dump, load
import numpy as np
import os


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # Create the generators for training

    path = "/home/luisals/test2/fixed_gamma/pearl_training2/"
    path_sims = "/lfstev/deepskies/luisals/"
    # path = "/mnt/beegfs/work/ati/pearl037/regression/test/fixed_gamma/"
    # path_sims = "/mnt/beegfs/work/ati/pearl037/"

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims, path=path_sims)

    training_particle_IDs = load(open(path + '../../pearl_training/training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path + '../../pearl_training/labels_training_set.pkl', 'rb'))

    val_particle_IDs = load(open(path + '../../pearl_training/validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path + '../../pearl_training/labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    dim = (51, 51, 51)
    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_tr)

    ######### TRAIN THE MODEL ################

    log_alpha = -3.5
    alpha = 10 ** log_alpha

    # Convolutional layers parameters

    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                       'kernel_regularizer': reg.l2_norm(alpha)}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l1_and_l21_group(alpha)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    # Regularization parameters + Cauchy likelihood

    reg_params = {'train_gamma': False, 'init_gamma': 0.2}

    # Train for 100 epochs

    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                          validation_generator=generator_validation, num_epochs=10, dim=generator_training.dim,
                          max_queue_size=10, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=0.0001,
                          save_summary=True, path_summary=path, validation_freq=1, train=True, compile=True)
