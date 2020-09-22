from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random
import sys


if __name__ == "__main__":

########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    num_epoch = sys.argv[1]
    saving_path = sys.argv[2]
    path_data = sys.argv[3]

    seed = 123
    np.random.seed(seed)
    python_random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Load data

    val_sim = "6"
    s = tn.SimulationPreparation([val_sim], path="/mnt/beegfs/work/ati/pearl037/")

    scaler = load(open(path_data + 'scaler_output.pkl', 'rb'))

    # val_particle_IDs = load(open(path_data + 'larger_validation_set.pkl', 'rb'))
    # val_labels_particle_IDS = load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))
    # dim = (75, 75, 75)
    # params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    # params_box = {'input_type': 'averaged', 'num_shells': 20}
    # generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
    #                                         shuffle=False, **params_val, **params_box)

    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    dim = (75, 75, 75)
    params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    params_box = {'input_type': 'averaged', 'num_shells': 20}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val, **params_box)


    ######### TRAIN THE MODEL ################

    alpha = 10**-4
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

    # pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
    # truth_rescaled = np.array([val_labels_particle_IDS[ID] for ID in val_particle_IDs])
    # h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    # true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    # np.save(saving_path + "small_val_predicted_sim_" + val_sim + "_epoch_" + num_epoch + ".npy", h_m_pred)
    # np.save(saving_path + "small_val_true_sim_" + val_sim + "_epoch_" + num_epoch + ".npy", true)

    dim = (75, 75, 75)
    params_box = {'input_type': 'averaged', 'num_shells': 20}

    params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr, **params_box)

    predt = Model.model.predict_generator(generator_training, use_multiprocessing=False, workers=0, verbose=1)
    truth_rescaledt = np.array([training_labels_particle_IDS[ID] for ID in training_particle_IDs])
    h_m_predt = scaler.inverse_transform(predt.reshape(-1, 1)).flatten()
    truet = scaler.inverse_transform(truth_rescaledt.reshape(-1, 1)).flatten()
    np.save(saving_path + "training_predicted_epoch_" + num_epoch + ".npy", h_m_predt)
    np.save(saving_path + "training_true_sim_epoch_" + num_epoch + ".npy", truet)
