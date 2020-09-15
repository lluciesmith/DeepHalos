import dlhalos_code.z0_data_processing as dp0
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random
import sys


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR VALIDATION #########

    num_epoch = sys.argv[1]
    saving_path = sys.argv[2]
    path_data = sys.argv[3]

    seed = 123
    np.random.seed(seed)
    python_random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Load data

    val_sim = "6"
    s = dp0.SimulationPreparation_z0([val_sim], path="/mnt/beegfs/work/ati/pearl037/")

    scaler = load(open(path_data + 'scaler_output.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'larger_validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))

    dim = (75, 75, 75)
    params_box = {'res_sim': 1667, 'rescale': True, 'dim': dim}
    generator_validation = dp0.DataGenerator_z0(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                                shuffle=False, batch_size=50,
                                                **params_box)

    ######### PREDICT FROM THE MODEL ################
    lr = 0.00005
    alpha = 10 ** -3

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

    weights = saving_path + "model/weights." + num_epoch + ".h5"
    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator={}, shuffle=True,
                          validation_generator=generator_validation, num_epochs=100, dim=generator_validation.dim,
                          max_queue_size=80, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=0.0001,
                          save_summary=False, path_summary=saving_path, validation_freq=1, train=False, compile=True,
                          initial_epoch=None, initialiser="Xavier_uniform")
    Model.model.load_weights(weights)

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
    truth_rescaled = np.array([val_labels_particle_IDS[ID] for ID in val_particle_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(saving_path + "predicted_sim_" + val_sim + "_epoch_" + num_epoch + ".npy", h_m_pred)
    np.save(saving_path + "true_sim_" + val_sim + "_epoch_" + num_epoch + ".npy", true)
