from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random
from dlhalos_code import loss_functions as lf
import tensorflow.keras as keras

if __name__ == "__main__":

########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    saving_path = "/mnt/beegfs/work/ati/pearl037/regression/full_mass_range/200k_random_training/9sims/Xavier" \
                  "/fixed_gamma/freeze_layer1/"

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
    # Added conv_6 in going from 31^3 input to 75^3 input

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l1_and_l21_group(alpha)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    weights = saving_path + "../model/weights.04.h5"
    Model = CNN.CNNCauchy(param_conv, param_fcc, lr=0.0001, model_type="regression", shuffle=True,
                          dim=generator_training.dim, training_generator=generator_training,
                          validation_generator=generator_validation, validation_freq=1,
                          num_epochs=100, verbose=1, seed=seed, init_gamma=0.2,
                          max_queue_size=80, use_multiprocessing=True,  workers=40, num_gpu=1,
                          save_summary=False,  path_summary=saving_path, compile=True, train=False,
                          load_weights=weights, initial_epoch=None,
                          alpha_mse=10**-4, load_mse_weights=False, use_mse_n_epoch=0, use_tanh_n_epoch=0,
                          initialiser="Xavier_uniform", train_gamma=False)
    model = Model.model
    w = model.get_weights()
    model.layers[1].trainable = False

    loss_c = lf.cauchy_selection_loss_fixed_boundary(gamma=0.2)
    optimiser = keras.optimizers.Adam(lr=Model.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
    model.compile(loss=loss_c, optimizer=optimiser)
    w2 = model.get_weights()

    assert all([np.allclose(w[i], w2[i]) for i in range(len(w))])

    callbacks_list, cbk = Model.get_callbacks()
    h = model.fit(generator_training, validation_data=generator_validation, initial_epoch=3, epochs=30,
                  callbacks=callbacks_list, verbose=1, shuffle=True, max_queue_size=80, use_multiprocessing=True,
                  workers=40)