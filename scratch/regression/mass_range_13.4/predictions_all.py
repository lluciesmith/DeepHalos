from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load, dump
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

    try:
        p_ids = load(open(path_data + '100k_particles_validation_set.pkl', 'rb'))
        labels_p_ids = load(open(path_data + '100k_labels_validation_set.pkl', 'rb'))
    except IOError:
        test_set = tn.InputsPreparation([val_sim], scaler_type="minmax", log_high_mass_limit=13.4,
                                        load_ids=False, shuffle=False, random_style="random", random_subset_all=100000,
                                        random_subset_each_sim=None, scaler_output=scaler, path=path_sims)
        dump(test_set.particle_IDs, open(path_data + '100k_particles_validation_set.pkl', 'wb'))
        dump(test_set.labels_particle_IDS, open(path_data + '100k_labels_validation_set.pkl', 'wb'))
        p_ids = test_set.particle_IDs
        labels_p_ids = test_set.labels_particle_IDS

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (75, 75, 75)}
    generator_validation = tn.DataGenerator(p_ids, labels_p_ids, s.sims_dic, shuffle=False, **params_val)


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
                          validation_generator={}, num_epochs=100, dim=generator_training.dim,
                          max_queue_size=80, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=0.0001,
                          save_summary=False, path_summary=saving_path, validation_freq=1, train=False, compile=True,
                          initial_epoch=None, initialiser="Xavier_uniform")
    Model.model.load_weights(weights)

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
    truth_rescaled = np.array([labels_p_ids[ID] for ID in p_ids])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(saving_path + "100k_predicted_sim_" + val_sim + "_epoch_" + num_epoch + ".npy", h_m_pred)
    np.save(saving_path + "100k_true_sim_" + val_sim + "_epoch_" + num_epoch + ".npy", true)

    if training_set is True:
        all_sims = ["%i" % i for i in np.arange(22)]
        all_sims.remove("3")
        s = tn.SimulationPreparation(all_sims, path="/mnt/beegfs/work/ati/pearl037/")

        training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
        training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))

        params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (75, 75, 75)}
        generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                              shuffle=False, **params_tr)

        predt = Model.model.predict_generator(generator_training, use_multiprocessing=False, workers=0, verbose=1)
        truth_rescaledt = np.array([training_labels_particle_IDS[ID] for ID in training_particle_IDs])
        h_m_predt = scaler.inverse_transform(predt.reshape(-1, 1)).flatten()
        truet = scaler.inverse_transform(truth_rescaledt.reshape(-1, 1)).flatten()
        np.save(saving_path + "training_predicted_epoch_" + num_epoch + ".npy", h_m_predt)
        np.save(saving_path + "training_true_epoch_" + num_epoch + ".npy", truet)

