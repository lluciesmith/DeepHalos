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

    params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (75, 75, 75)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_tr)


    ######### TRAIN THE MODEL ################

    alpha = 10**(-3.5)
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

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l1_and_l21_group(alpha)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    # Train for one epoch using MSE loss and the rest using a Cauchy loss

    weights = saving_path + "model/weights." + num_epoch + ".hf"
    Model = CNN.CNNCauchy(param_conv, param_fcc, lr=0.0001, model_type="regression", shuffle=True,
                          dim=generator_validation.dim, training_generator={},
                          validation_generator=generator_validation, validation_freq=1,
                          num_epochs=100, verbose=1, seed=seed, init_gamma=0.2,
                          max_queue_size=80, use_multiprocessing=True,  workers=40, num_gpu=1,
                          save_summary=False,  path_summary=saving_path,
                          compile=True, train=False,
                          load_weights=weights, initial_epoch=None,
                          alpha_mse=10**-4, load_mse_weights=True, use_mse_n_epoch=1, use_tanh_n_epoch=0
                          )

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
    truth_rescaled = np.array([val_labels_particle_IDS[ID] for ID in val_particle_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(saving_path + "predicted_" + val_sim + "_" + num_epoch + ".npy", h_m_pred)
    np.save(saving_path + "true_" + val_sim + "_" + num_epoch + ".npy", true)
