import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random


if __name__ == "__main__":

    pearl = sys.argv[1]
    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # np.random.seed(123)
    # python_random.seed(123)
    # tf.compat.v1.set_random_seed(1234)

    # Load data
    if pearl == "0":
        path_data = "/mnt/beegfs/work/ati/pearl037/regression/full_mass_range_51_3_fermi/"
        saving_path = "/mnt/beegfs/work/ati/pearl037/regression/full_mass_range_51_3_fermi/diff_seeds/"
        path_sims = "/mnt/beegfs/work/ati/pearl037/"

    else:
        path_data = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/" \
                    "lr_decay/cauchy_selec_bound_gamma_train_alpha/full_mass_range/"
        saving_path = path_data + "test/diff_seeds/"
        path_sims = "/lfstev/deepskies/luisals/"


    # path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set/random/"
    # num_sample = 50000
    # saving_path = path_data + str(num_sample) + "/"
    # scaler_output = load(open(saving_path + 'scaler_output.pkl', 'rb'))

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims, path=path_sims)

    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))

    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    dim = (51, 51, 51)
    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_tr)


    ######### TRAIN THE MODEL ################

    alpha = 10**(-3.5)
    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                       'kernel_regularizer': reg.l2_norm(alpha)}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  }
    # Added conv_6 in going from 31^3 input to 75^3 input

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l1_and_l21_group(alpha)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    reg_params = {'init_gamma': 0.2}

    # Train for one epoch using MSE loss

    Model = CNN.CNNCauchy(param_conv, param_fcc, lr=0.0001, model_type="regression",
                      dim=generator_training.dim, training_generator=generator_training,
                      validation_generator=generator_validation, num_epochs=60, validation_freq=1,
                      max_queue_size=10, use_multiprocessing=False,  workers=0, verbose=1, num_gpu=1,
                      save_summary=True, path_summary=saving_path, seed=None,
                      compile=True, train=True, load_weights=None,
                      load_mse_weights=False, use_mse_n_epoch=1, use_tanh_n_epoch=0, **reg_params)


