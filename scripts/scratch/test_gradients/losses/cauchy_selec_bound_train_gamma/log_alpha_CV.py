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

    # Load data

    path_data = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/"

    scaler_output = load(open(path_data + 'scaler_output_50000.pkl', "rb"))
    training_particle_IDs = load(open(path_data + 'training_set_50000.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set_50000.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=True, **params_val)


    ######### TRAIN THE MODEL ################

    path = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
           "/cauchy_selec_bound_gamma_train_alpha/l2_conv_l21_l1_dense/"

    log_alpha_grid = np.linspace(-3.1, -3.9, 5, endpoint=True)

    # for i, alpha in enumerate(log_alpha_grid):
    for log_alpha in log_alpha_grid:
        path_model = path + "log_alpha_" + str(log_alpha) + "/"
        alpha = 10**log_alpha

        # Convolutional layers parameters

        params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                           'kernel_regularizer': reg.l2_norm(alpha)}
        param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                      'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                      'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                      'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                      'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
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
                              workers=0, verbose=1, num_gpu=1, save_summary=True, path_summary=path_model,
                              compile=True, train=True, load_weights=None,
                              load_mse_weights=False, use_mse_n_epoch=1, use_tanh_n_epoch=0,
                              **reg_params)
