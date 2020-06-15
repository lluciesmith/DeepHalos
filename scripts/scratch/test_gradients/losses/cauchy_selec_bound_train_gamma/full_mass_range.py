import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import dump, load
import numpy as np
import os

if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # Create the generators for training

    path = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
           "/cauchy_selec_bound_gamma_train_alpha/full_mass_range/"

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)
    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    training_set = tn.InputsPreparation(train_sims, shuffle=True, scaler_type="minmax", return_rescaled_outputs=True,
                                        output_range=(-1, 1), load_ids=False, random_subset_each_sim=None,
                                        random_style="random", random_subset_all=50000)
    dump(training_set.particle_IDs, open(path + 'training_set.pkl', 'wb'))
    dump(training_set.labels_particle_IDS, open(path + 'labels_training_set.pkl', 'wb'))
    dump(training_set.scaler_output, open(path + 'scaler_output.pkl', 'wb'))

    v_set = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False, shuffle=True,
                                 random_style="random", random_subset_all=5000, random_subset_each_sim=None,
                                 scaler_output=training_set.scaler_output)
    dump(v_set.particle_IDs, open(path + 'validation_set.pkl', 'wb'))
    dump(v_set.labels_particle_IDS, open(path + 'labels_validation_set.pkl', 'wb'))

    dim = (51, 51, 51)
    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_validation = tn.DataGenerator(v_set.particle_IDs, v_set.labels_particle_IDS, s.sims_dic,
                                            shuffle=True, **params_val)


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

    reg_params = {'init_gamma': 0.2}

    # Train for 100 epochs

    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", dim=generator_training.dim,
                          training_generator=generator_training, validation_generator=generator_validation,
                          num_epochs=60, validation_freq=1, lr=0.0001, max_queue_size=10,
                          use_multiprocessing=False,
                          workers=0, verbose=1, num_gpu=1, save_summary=True, path_summary=path,
                          compile=True, train=True, load_weights=None,
                          load_mse_weights=False, use_mse_n_epoch=1, use_tanh_n_epoch=0,
                          **reg_params)
