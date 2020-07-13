from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
from tensorflow.keras import backend as k
from dlhalos_code import loss_functions as lf
import numpy as np
import tensorflow as tf
import random as python_random
import sys


def get_data():
    path_sims = "/mnt/beegfs/work/ati/pearl037/"
    all_sims = ["6"]
    s = tn.SimulationPreparation(all_sims, path=path_sims)

    path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set/9sims/random/200k/"
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    dim = (75, 75, 75)
    params_val = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)
    return generator_validation


if __name__ == "__main__":

    saving_path = "/mnt/beegfs/work/ati/pearl037/regression/full_mass_range/test_init_activation/"

    g_val = get_data()

    seed = 123
    np.random.seed(seed)
    python_random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    ######### INITIALIZE THE MODEL ################

    initialiser = sys.argv[1]
    activation_choice = sys.argv[2]

    if activation_choice == "Lrelu":
        p_act = {'activation': "linear", 'relu': True}
    elif activation_choice == "tanh":
        p_act = {'activation': "tanh", 'relu': False}
    elif activation_choice == "linear":
        p_act = {'activation': "linear", 'relu': False}
    else:
        raise ValueError("Choose the correct option for the activation")

    alpha = 10**(-3.5)
    params_all_conv = {'strides': 1, 'padding': 'same', 'bn': False, 'kernel_regularizer': reg.l2_norm(alpha)}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv, **p_act},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act}
                  }

    params_all_fcc = {'bn': False, 'kernel_regularizer': reg.l1_and_l21_group(alpha)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc, **p_act},
                 'dense_2': {'neurons': 128, **params_all_fcc, **p_act},
                 'last': {}}

    Model = CNN.CNNCauchy(param_conv, param_fcc, lr=0.0001, model_type="regression", shuffle=True,
                          dim=(75, 75, 75), training_generator={},
                          validation_generator={}, validation_freq=1,
                          num_epochs=100, verbose=1, seed=seed, init_gamma=0.2,
                          max_queue_size=80, use_multiprocessing=True,  workers=40, num_gpu=1,
                          save_summary=False,  path_summary=saving_path, compile=True, train=False,
                          load_weights=None, initial_epoch=None,
                          alpha_mse=10**-4, load_mse_weights=False, use_mse_n_epoch=0, use_tanh_n_epoch=0,
                          initialiser=initialiser, train_gamma=True
                          )

    w = Model.model.trainable_weights
    out = Model.model.output
    loss_c = lf.cauchy_selection_loss_fixed_boundary_trainable_gamma(Model.model.layers[-1])
    # loss_mse = tf.keras.losses.mean_squared_error()

    g = g_val[0]
    labels = np.float32(g[1].reshape(len(g[1]), 1))
    l = loss_c(labels, out)
    gradients = k.gradients(l, w)

    sess = k.get_session()
    evaluated_gradients = sess.run(gradients, feed_dict={Model.model.input: np.float32(g[0])})
    np.save(saving_path + "gradients_" + initialiser + "_" + activation_choice + "_epoch0.npy", evaluated_gradients)
    np.save(saving_path + "weights_" + initialiser + "_" + activation_choice + "_epoch0.npy", Model.model.get_weights())


