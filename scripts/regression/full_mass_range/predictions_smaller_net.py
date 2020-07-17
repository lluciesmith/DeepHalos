from dlhalos_code import CNN
import sys
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random


if __name__ == "__main__":
    # two script inputs

    num_epoch = sys.argv[1]
    saving_path = sys.argv[2]

    # num_epoch = "10"
    # saving_path = "/mnt/beegfs/work/ati/pearl037/regression/full_mass_range/200k_random_training/9sims/"

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    val_sim = "6"
    s = tn.SimulationPreparation([val_sim], path="/mnt/beegfs/work/ati/pearl037/")

    path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set/9sims/random/200k/"
    val_particle_IDs = load(open(path_data + 'larger_validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))
    scaler = load(open(path_data + 'scaler_output.pkl', 'rb'))

    # Create the generators for training

    params_tr = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (75, 75, 75)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_tr)


    ######### TRAIN THE MODEL ################

    alpha = 10**(-3.5)
    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                       'kernel_regularizer': reg.l2_norm(alpha)
                       }
    param_conv = {'conv_1': {'num_kernels': 16, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_2': {'num_kernels': 16, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  # 'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  # 'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l1_and_l21_group(10**-3.5)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    # Train for one epoch using MSE loss and the rest using a Cauchy loss

    weights = saving_path + "model/weights." + num_epoch + ".h5"
    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator={}, shuffle=True,
                          validation_generator=generator_validation, metrics=[CNN.likelihood_metric], num_epochs=30,
                          dim=(75, 75, 75), initialiser="Xavier_uniform", max_queue_size=10, use_multiprocessing=False,
                          workers=0, verbose=1, num_gpu=1, lr=0.0001, save_summary=True, path_summary=saving_path,
                          validation_freq=1, train=False, compile=True, initial_epoch=14, seed=None)

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
    truth_rescaled = np.array([val_labels_particle_IDS[ID] for ID in val_particle_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(saving_path + "predicted_sim_" + val_sim + "_epoch_" + num_epoch + ".npy", h_m_pred)
    np.save(saving_path + "true_sim_" + val_sim + "_epoch_" + num_epoch + ".npy", true)
