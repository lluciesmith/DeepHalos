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

    initialiser = "Xavier_uniform"
    activation_choice = "Lrelu"

    if activation_choice == "Lrelu":
        p_act = {'activation': "linear", 'relu': True}
    elif activation_choice == "tanh":
        p_act = {'activation': "tanh", 'relu': False}
    elif activation_choice == "linear":
        p_act = {'activation': "linear", 'relu': False}
    else:
        raise ValueError("Choose the correct option for the activation")

    alpha = 10**(-3.5)
    params_all_conv = {'strides': 1, 'padding': 'same', 'bn': False,
                       # 'kernel_regularizer': reg.l2_norm(alpha)
                       }
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv, **p_act},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv, **p_act}
                  }

    params_all_fcc = {'bn': False,
                      #'kernel_regularizer': reg.l1_and_l21_group(alpha)
                      }
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc, **p_act},
                 'dense_2': {'neurons': 128, **params_all_fcc, **p_act},
                 'last': {}}

    # Model = CNN.CNNCauchy(param_conv, param_fcc, lr=0.0001, model_type="regression", shuffle=True,
    #                       dim=(75, 75, 75), training_generator={},
    #                       validation_generator={}, validation_freq=1,
    #                       num_epochs=100, verbose=1, seed=seed, init_gamma=0.2,
    #                       max_queue_size=80, use_multiprocessing=True,  workers=40, num_gpu=1,
    #                       save_summary=False,  path_summary=saving_path, compile=True, train=False,
    #                       load_weights=None, initial_epoch=None,
    #                       alpha_mse=10**-4, load_mse_weights=False, use_mse_n_epoch=0, use_tanh_n_epoch=0,
    #                       initialiser=initialiser, train_gamma=True
    #                       )

    Model = CNN.CNN(param_conv, param_fcc, lr=0.0001, model_type="regression", shuffle=True,
                          dim=(75, 75, 75), training_generator={}, loss="mse",
                          validation_generator={}, validation_freq=1,
                          num_epochs=100, verbose=1, seed=seed, initialiser=initialiser,
                          max_queue_size=80, use_multiprocessing=True,  workers=40, num_gpu=1,
                          save_summary=False,  path_summary=saving_path, compile=True, train=False
                          )

    g = g_val[0]
    labels = np.float32(g[1].reshape(len(g[1]), 1))

    w = Model.model.trainable_weights
    out = Model.model.output
    # loss_func = lf.cauchy_selection_loss_fixed_boundary_trainable_gamma(Model.model.layers[-1])
    loss_func = tf.keras.losses.mean_squared_error

    l = loss_func(labels, out)
    gradients = k.gradients(l, w)

    sess = k.get_session()
    evaluated_gradients = sess.run(gradients, feed_dict={Model.model.input: np.float32(g[0])})
    np.save(saving_path + "mse_gradients_" + initialiser + "_" + activation_choice + "_epoch0.npy", evaluated_gradients)
    np.save(saving_path + "mse_weights_" + initialiser + "_" + activation_choice + "_epoch0.npy",
            Model.model.get_weights())

    # tr_weights = [w for layer in Model.model.layers for w in layer.trainable_weights
    #               if layer.trainable and ('bias' not in w.op.name)]
    # tr_g = Model.model.optimizer.get_gradients(Model.model.total_loss, tr_weights)
    # get_g = k.function(inputs=[Model.model._feed_inputs, Model.model._feed_targets], outputs=tr_g)
    # output_grad = get_g([np.float32(g[0]), labels])
    #
    # np.save(saving_path + "optimizer_gradients_" + initialiser + "_" + activation_choice + "_epoch0.npy", output_grad)
    # np.save(saving_path + "optimizer_weights_" + initialiser + "_" + activation_choice + "_epoch0.npy",
    #         Model.model.get_weights())




# # PLots
# import matplotlib.pyplot as plt
#
# p = "/Users/lls/Documents/deep_halos_files/full_mass_range/test_init_activation/"
#
# init = ["Xavier_uniform", "Gaussian"]
# titles = ["Xavier", "Gaussian"]
# act = ["Lrelu", "tanh", "linear"]
#
# conv_layers = [0, 2, 4, 6, 8, 10]
# dense_layers = [12, 14, 16]
# conv_layers_g = [0, 1, 2, 3, 4, 5]
# dense_layers_g = [6, 7, 8]
#
# for i, init_i in enumerate(init):
#     title = "MSE loss, " + titles[i]
#
#     for layers in [conv_layers, dense_layers]:
#         if layers == conv_layers:
#             positions = [1, 2, 3, 4, 5, 6]
#             label="Convolutional layers"
#             layers_g = conv_layers_g
#
#             title_plot = p + "plots/" + title + "_conv_layers.png"
#         else:
#             positions = [1, 2, 3]
#             label = "Fully-connected layers"
#             layers_g = dense_layers_g
#
#             title_plot = p + "plots/" + title + "_dense_layers.png"
#
#         f, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
#
#         w = np.load(p + "optimizer_weights_" + init_i + "_" + act[0] + "_epoch0.npy", allow_pickle=True)
#         axes[0, 1].violinplot([w[k].flatten() for k in layers], positions=positions)
#         title = titles[i]
#         axes[0, 1].set_title(title)
#
#         axes[0,0].set_visible(False)
#         axes[0, 2].set_visible(False)
#
#         for j, act_i in enumerate(act):
#             g = np.load(p + "optimizer_gradients_" + init_i + "_" + act_i + "_epoch0.npy", allow_pickle=True)
#             axes[1, j].violinplot([g[k].flatten() for k in layers_g], positions=positions)
#             axes[1, j].set_title(act_i)
#
#         # y_lims = np.array([axes[1, 0].get_ylim(), axes[1, 1].get_ylim()]).flatten()
#         # [axes[y_lims.min(), y_lims.max()].set_ylim(axes[1, jj]) for jj in [0, 1, 2]]
#         axes[1, 1].set_xlabel(label)
#         axes[0, 1].set_ylabel("Weights")
#         axes[1, 0].set_ylabel("Gradients")
#
#         # [axes[1, jj].set_yticks([]) for jj in [1, 2]]
#
#         # [axes[ii, jj].set_xlabel("Fully-connected layers") for ii in [2, 3] for jj in [0, 1, 2]]
#         # [axes[ii, 0].set_ylabel("Gradients") for ii in [1, 3]]
#         plt.subplots_adjust(top=0.9, left=0.14)
#         # plt.savefig(title_plot)
#


