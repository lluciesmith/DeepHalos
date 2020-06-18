import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from dlhalos_code import evaluation as evalu
from pickle import dump, load
import numpy as np
import os
import gc
import numpy as np
import tensorflow as tf
import random as python_random

pearl = True

#for i in range(2):
#seed_value = 123
np.random.seed(123)
python_random.seed(123)
tf.compat.v1.set_random_seed(1234)

if pearl:
    path = "/mnt/beegfs/work/ati/pearl037/regression/test/"
    path_sims = "/mnt/beegfs/work/ati/pearl037/"
else:
    path = "/home/luisals/test/"
    path_sims = "/lfstev/deepskies/luisals/"

all_sims = ["0", "1", "2", "4", "5", "6"]
s = tn.SimulationPreparation(all_sims, path=path_sims)
train_sims = all_sims[:-1]
val_sim = all_sims[-1]

p_ids = load(open(path + 'training_set.pkl', 'rb'))
l_ids = load(open(path + 'labels_training_set.pkl', 'rb'))
v_ids = load(open(path + 'validation_set.pkl', 'rb'))
l_v_ids = load(open(path + 'labels_validation_set.pkl', 'rb'))

dim = (51, 51, 51)
params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
generator_training = tn.DataGenerator(p_ids, l_ids, s.sims_dic, shuffle=False, **params_tr)
generator_validation = tn.DataGenerator(v_ids, l_v_ids, s.sims_dic, shuffle=False, **params_tr)

# Convolutional layers parameters

log_alpha = -3.5
alpha = 10 ** log_alpha

params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                   'kernel_regularizer': reg.l2_norm(alpha)}
param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
              'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
              }

# Dense layers parameters

params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                  'kernel_regularizer': reg.l2_norm(alpha)}
param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
             'last': {}}

# Regularization parameters + Cauchy likelihood

reg_params = {'init_gamma': 0.2}

# for i in range(10):
#     np.random.seed(123)
#     python_random.seed(123)
#     tf.compat.v1.set_random_seed(1234)
#
#     try:
#         os.mkdir(path + 'run_' + str(i))
#     except:
#         pass
#     path1 = path + 'run_' + str(i) + '/'

# Train for 100 epochs

Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", dim=generator_training.dim,
                      training_generator=generator_training, validation_generator=generator_validation,
                      num_epochs=20, validation_freq=1, lr=0.0001, max_queue_size=10,
                      use_multiprocessing=False,
                      workers=0, verbose=1, num_gpu=1, save_summary=True, path_summary=path,
                      compile=True, train=False, load_weights=None,
                      load_mse_weights=True, use_mse_n_epoch=10, use_tanh_n_epoch=0,
                      **reg_params)

# Model1 = CNN.CNN(param_conv, param_fcc, model_type="regression", dim=generator_training.dim,
#                 training_generator=generator_training, validation_generator=generator_validation,
#                 num_epochs=5, validation_freq=1, lr=0.0001, max_queue_size=10,
#                 use_multiprocessing=False, seed=1234,
#                 workers=0, verbose=1, num_gpu=1, save_summary=True, path_summary=path,
#                 compile=True, train=True,
#                 # load_weights=None, load_mse_weights=False, use_mse_n_epoch=10, use_tanh_n_epoch=0,
#                 # reg_params
#                 )
#
#
# Model2 = CNN.CNN(param_conv, param_fcc, model_type="regression", dim=generator_training.dim,
#                 training_generator=generator_training, validation_generator=generator_validation,
#                 num_epochs=5, validation_freq=1, lr=0.0001, max_queue_size=10,
#                 use_multiprocessing=False, seed=1234,
#                 workers=0, verbose=1, num_gpu=1, save_summary=True, path_summary=path,
#                 compile=True, train=True,
#                 # load_weights=None, load_mse_weights=False, use_mse_n_epoch=10, use_tanh_n_epoch=0,
#                 # reg_params
#                 )

# del tn
# from dlhalos_code import data_processing as tn
# import importlib
# importlib.reload(tn)
# s = tn.SimulationPreparation(all_sims, path=path_sims)
# generator_training = tn.DataGenerator(p_ids, l_ids, s.sims_dic, shuffle=False, **params_tr)
# generator_validation = tn.DataGenerator(v_ids, l_v_ids, s.sims_dic, shuffle=False, **params_tr)
#
# Model = CNN.CNN(param_conv, param_fcc, model_type="regression", dim=generator_training.dim,
#                 training_generator=generator_training, validation_generator=generator_validation,
#                 num_epochs=5, validation_freq=1, lr=0.0001, max_queue_size=10,
#                 use_multiprocessing=False,
#                 workers=0, verbose=1, num_gpu=1, save_summary=True, path_summary=path,
#                 compile=True, train=True,
#                 # load_weights=None, load_mse_weights=False, use_mse_n_epoch=10, use_tanh_n_epoch=0,
#                 # reg_params
#                 )

# del Model
# gc.collect()



# training_set = tn.InputsPreparation(train_sims, shuffle=True, scaler_type="minmax",
#                                     return_rescaled_outputs=True, output_range=(-1, 1), load_ids=False,
#                                     random_style="random", random_subset_all=1000, random_subset_each_sim=1000,
#                                     path=path_sims)
#
# dump(training_set.particle_IDs, open(path + 'training_set.pkl', 'wb'))
# dump(training_set.labels_particle_IDS, open(path + 'labels_training_set.pkl', 'wb'))
# dump(training_set.scaler_output, open(path + 'scaler_output.pkl', 'wb'))
#
# v_set = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False, shuffle=True,
#                              random_style="random", random_subset_all=200, random_subset_each_sim=None,
#                              scaler_output=training_set.scaler_output, path=path_sims)
# dump(v_set.particle_IDs, open(path + 'validation_set.pkl', 'wb'))
# dump(v_set.labels_particle_IDS, open(path + 'labels_validation_set.pkl', 'wb'))
#
# dim = (51, 51, 51)
# params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
# generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
#                                       shuffle=True, **params_tr)
#
# params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
# generator_validation = tn.DataGenerator(v_set.particle_IDs, v_set.labels_particle_IDS, s.sims_dic,
#                                         shuffle=True, **params_val)