import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import regularizers as reg
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
           "/cauchy_selec_bound_gamma_cv_alpha/"

    # Define model

    alpha_grid = [10**-j for j in np.arange(2, 8).astype("float")]

    for alpha in alpha_grid:
        path_model = path + "alpha_" + str(alpha) + "/"

        if alpha > 10**-5:
            os.mkdir(path_model)
            os.mkdir(path_model + "model/")
            resume_training = False
        else:
            resume_training = True

        # Here we do not train alpha but we do a grid search

        conv_l2 = reg.l2_norm(alpha)
        dense_l21_l1 = reg.l1_and_l21_group(alpha)

        params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                           'kernel_regularizer': conv_l2}
        param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                      'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                      'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                      'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                      'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                      }

        params_all_fcc = {'bn': False, 'dropout': 0.4, 'activation': "linear", 'relu': True,
                          'kernel_regularizer': dense_l21_l1}
        param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc}, 'last': {}}

        # Train for 60 epochs

        if resume_training:
            Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression",
                                  training_generator=generator_training, validation_generator=generator_validation,
                                  validation_freq=1, lr=0.0001, max_queue_size=10,
                                  use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, save_summary=True,
                                  path_summary=path_model, compile=True, train=True,  load_mse_weights=True,
                                  num_epochs=60, initial_epoch=30, load_weights=path_model + "model/weights.30.hdf5")
        else:
            Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression",
                                  training_generator=generator_training, validation_generator=generator_validation,
                                  num_epochs=60, validation_freq=1, lr=0.0001, max_queue_size=10,
                                  use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, save_summary=True,
                                  path_summary=path_model, compile=True,
                                  train=True, load_mse_weights=False)