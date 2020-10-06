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

    # saving_path = "/mnt/beegfs/work/ati/pearl037/regression/mass_range_13.4/random_9sims/"
    # path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set_13.4/20sims/random/200k/"

    num_epoch = sys.argv[1]
    saving_path = sys.argv[2]
    path_data = sys.argv[3]
    training_set = False

    seed = 123
    np.random.seed(seed)
    python_random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Load data

    val_sim = "6"
    path_sims = "/mnt/beegfs/work/ati/pearl037/"
    s = tn.SimulationPreparation([val_sim], path="/mnt/beegfs/work/ati/pearl037/")
    scaler = load(open(path_data + 'scaler_output.pkl', 'rb'))

    test_set = tn.InputsPreparation([val_sim], scaler_type="minmax", log_high_mass_limit=13.4,
                                    load_ids=False, shuffle=False, random_style=None, random_subset_all=None,
                                    random_subset_each_sim=None, scaler_output=scaler, path=path_sims)

    params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (75, 75, 75)}
    generator_validation = tn.DataGenerator(test_set.particle_IDs, test_set.labels_particle_IDS, s.sims_dic,
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

    weights = saving_path + "model/weights." + num_epoch + ".h5"
    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator={}, shuffle=True,
                          validation_generator=generator_validation, num_epochs=100, dim=generator_validation.dim,
                          max_queue_size=80, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=0.0001,
                          save_summary=False, path_summary=saving_path, validation_freq=1, train=False, compile=True,
                          initial_epoch=None, initialiser="Xavier_uniform")
    Model.model.load_weights(weights)

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
    truth_rescaled = np.array([test_set.labels_particle_IDS[ID] for ID in test_set.particle_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(saving_path + "all_predicted_sim_" + val_sim + "_epoch_" + num_epoch + "_2.npy", h_m_pred)
    np.save(saving_path + "all_true_sim_" + val_sim + "_epoch_" + num_epoch + "_2.npy", true)
