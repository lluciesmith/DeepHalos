from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random
import sys
import os
from pickle import dump


if __name__ == "__main__":

########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    saving_path = "/mnt/beegfs/work/ati/pearl037/regression/mass_range_13.4/random_20sims_200K/lr_10-5/diff_training/"

    seed = 123
    np.random.seed(seed)
    python_random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Load data

    path_sims = "/mnt/beegfs/work/ati/pearl037/"
    all_sims = ["%i" % i for i in np.arange(22)]
    all_sims.remove("3")
    all_sims.remove("6")
    all_sims.append("6")
    s = tn.SimulationPreparation(all_sims, path=path_sims)

    training_set = tn.InputsPreparation(all_sims[:-1], shuffle=True, scaler_type="minmax", return_rescaled_outputs=True,
                                        output_range=(-1, 1), log_high_mass_limit=13.4,
                                        load_ids=False, random_subset_each_sim=None,
                                        random_style="random", random_subset_all=200000,
                                        path=path_sims)

    dump(training_set.particle_IDs, open(saving_path + 'training_set.pkl', 'wb'))
    dump(training_set.labels_particle_IDS, open(saving_path + 'labels_training_set.pkl', 'wb'))
    dump(training_set.scaler_output, open(saving_path + 'scaler_output.pkl', 'wb'))

    v_set = tn.InputsPreparation([all_sims[-1]], scaler_type="minmax", load_ids=False, shuffle=True,
                                 random_style="random", log_high_mass_limit=13.4,
                                 random_subset_all=5000, random_subset_each_sim=1000000,
                                 scaler_output=training_set.scaler_output, path=path_sims)
    dump(v_set.particle_IDs, open(saving_path + 'validation_set.pkl', 'wb'))
    dump(v_set.labels_particle_IDS, open(saving_path + 'labels_validation_set.pkl', 'wb'))

    # Create the generators for training

    dim = (75, 75, 75)
    params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)
    params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_validation = tn.DataGenerator(v_set.particle_IDs, v_set.labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)


    ######### TRAIN THE MODEL ################

    alpha = 10**-2.5
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
                          use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=0.00001, save_summary=True,
                          path_summary=saving_path, validation_freq=1, train=True, compile=True,
                          initial_epoch=None,
                          seed=seed)

