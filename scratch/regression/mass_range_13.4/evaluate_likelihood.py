from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random
import sys
import os


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    saving_path_ = "/mnt/beegfs/work/ati/pearl037/regression/mass_range_13.4/random_20sims_200K/lr_5e-5/"

    log_alpha = sys.argv[1]
    saving_path = saving_path_ + "alpha_" + str(log_alpha) + "/"

    seed = 123
    np.random.seed(seed)
    python_random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Load data

    val_sim = "6"
    s = tn.SimulationPreparation([val_sim], path="/mnt/beegfs/work/ati/pearl037/")

    path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set_13.4/20sims/random/200k/"
    scaler = load(open(path_data + 'scaler_output.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'larger_validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))
    # val_small = load(open(path_data + 'validation_set.pkl', 'rb'))
    # val_small_labels = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    dim = (75, 75, 75)
    params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)


    ######### TRAIN THE MODEL ################

    alpha = 10**float(log_alpha)
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

    lr = 0.00005
    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator={}, shuffle=True,
                          validation_generator=generator_validation, num_epochs=100, dim=generator_validation.dim,
                          max_queue_size=80, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=lr,
                          save_summary=False, path_summary=saving_path, validation_freq=1, train=False, compile=True,
                          initial_epoch=None, initialiser="Xavier_uniform", metrics=[CNN.likelihood_metric])

    epochs = ["%02d" % num for num in np.arange(1, 19)]
    loss = []
    lik = []
    for num_epoch in epochs:
        weights = saving_path + "model/weights." + num_epoch + ".h5"
        Model.model.load_weights(weights)
        scores = Model.model.evaluate_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
        loss.append(scores[0])
        lik.append(scores[1])

    np.save(saving_path + "loss_larger_val_set.npy", loss)
    np.save(saving_path + "likelihood_larger_val_set.npy", lik)
