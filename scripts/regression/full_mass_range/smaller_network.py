from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    saving_path = "/mnt/beegfs/work/ati/pearl037/regression/full_mass_range/200k_random_training/9sims/Xavier" \
                  "/fixed_gamma/smaller_net_glob_avg/"

    seed = 123
    np.random.seed(seed)
    python_random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # Load data

    path_sims = "/mnt/beegfs/work/ati/pearl037/"
    all_sims = ["0", "1", "2", "4", "5", "7", "8", "9", "10", "6"]
    s = tn.SimulationPreparation(all_sims, path=path_sims)

    path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set/9sims/random/200k/"
    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    dim = (75, 75, 75)
    params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)
    params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)


    ######### TRAIN THE MODEL ################

    # alpha = 10**(-3.5)
    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                       'kernel_regularizer': reg.l2_norm(10**(-3.5))
                       }

    param_conv = {'conv_1': {'num_kernels': 16, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_2': {'num_kernels': 16, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  }

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l1_and_l21_group(10**-4.5),
                      'dropout': 0.4}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc},
                 'dense_2': {'neurons': 128, **params_all_fcc},
                 'dense_3': {'neurons': 20, **params_all_fcc},
                 'last': {}}

    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                          shuffle=True, validation_generator=generator_validation, metrics=[CNN.likelihood_metric],
                          num_epochs=30, dim=generator_training.dim,
                          initialiser="Xavier_uniform", max_queue_size=80,
                          use_multiprocessing=False, workers=0, verbose=1, num_gpu=1,
                          lr=0.0001, save_summary=True,
                          path_summary=saving_path, validation_freq=1, train=False, compile=True,
                          initial_epoch=6, #weights=saving_path + "model/weights.06.h5",
                          seed=seed, global_average=True)

    Model.model.load_weights(saving_path + "model/weights.06.h5")
    c = Model.get_callbacks(layer_loss=Model.model.layers[-1])[0]
    Model.model.fit_generator(generator=Model.training_generator, validation_data=Model.validation_generator,
                                          use_multiprocessing=Model.use_multiprocessing, workers=Model.workers,
                                          max_queue_size=Model.max_queue_size, initial_epoch=Model.initial_epoch,
                                          verbose=Model.verbose, epochs=Model.num_epochs, shuffle=Model.shuffle,
                                          callbacks=c, validation_freq=Model.val_freq,
                                          validation_steps=Model.validation_steps, steps_per_epoch=5)


