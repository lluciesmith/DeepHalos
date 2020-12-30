import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from dlhalos_code import evaluation as evalu
from pickle import dump, load
import numpy as np
import os

########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

# Create the generators for training


# path = "/lfstev/deepskies/luisals/regression/test_pearl/"
# path_sims = "/lfstev/deepskies/luisals/"

path = "/mnt/beegfs/work/ati/pearl037/regression/test_pearl/"
path_sims = "/mnt/beegfs/work/ati/pearl037/"

all_sims = ["0", "1", "2", "4", "5", "6"]
s = tn.SimulationPreparation(all_sims, path=path_sims)
val_sim = all_sims[-1]

for sim in all_sims:
    train_sims = [sim]
    try:
        os.mkdir(path + 'sim_' + sim)
    except:
        pass
    training_set = tn.InputsPreparation(train_sims, shuffle=False, scaler_type="minmax", return_rescaled_outputs=True,
                                        output_range=(-1, 1), load_ids=False,
                                        random_style="random", random_subset_all=100,
                                        random_subset_each_sim=None,
                                        # random_style="uniform", random_subset_each_sim=1000000, num_per_mass_bin=10000,
                                        path=path_sims)
    dim = (31, 31, 31)
    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
                                          shuffle=False, **params_tr)
    dump(training_set.particle_IDs, open(path + 'sim_' + sim + '/particle_ids.pkl', 'wb'))
    dump(training_set.labels_particle_IDS, open(path + 'sim_' + sim + '/labels_particle_ids.pkl', 'wb'))
    dump(generator_training[0], open(path + 'sim_' + sim + '/generator_trainings.pkl', 'wb'))

# sims = ["training_simulation", "reseed1_simulation", "reseed2_simulation", "reseed4_simulation",
#         "reseed5_simulation", "reseed6_simulation"]
# sims_num = ["0", "1", "2", "4", "5", "6"]
# path_sims = "/Users/lls/Desktop/data_transfer/"
# path = "/Users/lls/Desktop/PEARL/"
#
# for i, sim in enumerate(sims_num):
#     s = tn.SimulationPreparation([sim], path=path_sims)
#
#     p_ids = load(open(path + 'sim_' + sim + '/particle_ids.pkl', 'rb'))
#     l_ids = load(open(path + 'sim_' + sim + '/labels_particle_ids.pkl', 'rb'))
#     gen = load(open(path + 'sim_' + sim + '/generator_trainings.pkl', 'rb'))
#
#     params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
#     generator_training = tn.DataGenerator(p_ids, l_ids, s.sims_dic,
#                                           shuffle=False, **params_tr)
#     np.allclose(gen[0], generator_training[0][0])
#     np.allclose(gen[1], generator_training[0][1])
#     del s


# params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
# generator_validation = tn.DataGenerator(v_set.particle_IDs, v_set.labels_particle_IDS, s.sims_dic,
#                                         shuffle=True, **params_val)
#
# ######### TRAIN THE MODEL ################
#
# log_alpha = -3.5
# alpha = 10 ** log_alpha
#
# # Convolutional layers parameters
#
# params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
#                    'kernel_regularizer': reg.l2_norm(alpha)}
# param_conv = {'conv_1': {'num_kernels': 2, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
#               'conv_2': {'num_kernels': 2, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
#               'conv_3': {'num_kernels': 2, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
#               'conv_4': {'num_kernels': 2, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
#               'conv_5': {'num_kernels': 2, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
#               }
#
# # Dense layers parameters
#
# params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
#                   'kernel_regularizer': reg.l1_and_l21_group(alpha)}
# param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
#              'last': {}}
#
# # Regularization parameters + Cauchy likelihood
#
# reg_params = {'init_gamma': 0.2}
#
# Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", dim=generator_training.dim,
#                       training_generator=generator_training, validation_generator=generator_validation,
#                       num_epochs=10, validation_freq=1, lr=0.0001, max_queue_size=10,
#                       use_multiprocessing=False,
#                       workers=0, verbose=1, num_gpu=1, save_summary=True, path_summary=path,
#                       compile=True, train=True, load_weights=None,
#                       load_mse_weights=False, use_mse_n_epoch=1, use_tanh_n_epoch=0,
#                       **reg_params)
