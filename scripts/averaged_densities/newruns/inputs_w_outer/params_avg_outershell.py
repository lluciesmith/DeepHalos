from pickle import load
import numpy as np
from dlhalos_code import custom_regularizers as reg

seed = 123
path = "/share/hypatia/lls/newdlhalos/spherically_averaged_Mmin11/w_outer_shells/"

# Data parameters
path_sims = "/share/hypatia/lls/simulations/dlhalos_sims/"
all_sims = ["%i" % i for i in np.arange(25)]
all_sims.remove("3")
val_sim = ["6"]
test_sim = ["6", "22", "23", "24"]
train_sims = list(np.array(all_sims)[np.where(~np.in1d(all_sims, test_sim) & ~np.in1d(all_sims, val_sim))[0]])
path_data = "/share/hypatia/lls/newdlhalos/training_data/"

# Load training/testing particle IDs
training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))
scaler = load(open(path_data + 'scaler_output.pkl', 'rb'))
test_particle_IDs = load(open(path_data + 'test_set.pkl', 'rb'))
test_labels_particle_IDS = load(open(path_data + 'labels_test_set.pkl', 'rb'))

# inputs parameters
dim = (75, 75, 75)
params_box = {'input_type': 'averaged_wouter', 'num_shells': 20}
params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}

# Model params
lr = 0.00005
log_alpha = -4.3
alpha = 10 ** log_alpha
saving_path = path + "/log_alpha_" + str(log_alpha) + "/"

# Convolutional layers parameters
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

# Dense layers parameters

params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                  'kernel_regularizer': reg.l1_and_l21_group(alpha)}
param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
             'last': {}}

