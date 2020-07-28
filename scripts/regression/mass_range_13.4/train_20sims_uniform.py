from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random


if __name__ == "__main__":

########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    saving_path = "/mnt/beegfs/work/ati/pearl037/regression/mass_range_13.4/uniform_20sims_5k_bins/"

    seed = 123
    np.random.seed(seed)
    python_random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Load data

    path_sims = "/mnt/beegfs/work/ati/pearl037/"
    all_sims = ["%i" % i for i in np.arange(22)]
    all_sims.remove("3")
    s = tn.SimulationPreparation(all_sims, path=path_sims)

    path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set_13.4/20sims/uniform/5k_in_each_80bins/"
    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    dim = (75, 75, 75)
    params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)
    params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)


    ######### TRAIN THE MODEL ################

    alpha = 10**-3
    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                       'kernel_regularizer': reg.l2_norm(alpha)
                       }
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }
    # Added conv_6 in going from 31^3 input to 75^3 input

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l1_and_l21_group(alpha)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                          shuffle=True, validation_generator=generator_validation, num_epochs=30,
                          metrics=[CNN.likelihood_metric],
                          steps_per_epoch=len(generator_training), validation_steps=len(generator_validation),
                          dim=generator_training.dim, initialiser="Xavier_uniform", max_queue_size=8,
                          use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=0.0001, save_summary=True,
                          path_summary=saving_path, validation_freq=1, train=True, compile=True,
                          initial_epoch=None,
                          seed=seed)

