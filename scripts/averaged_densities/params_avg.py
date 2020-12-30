from pickle import load
import numpy as np
from dlhalos_code import custom_regularizers as reg

seed = 123
num_epoch_testing = "32"
test_training = False
val_sim = "6"

# regularization parameters

log_alpha = -4.3
alpha = 10 ** log_alpha
log_alpha_str = str(log_alpha)
log_alpha_int = float(log_alpha)
saving_path = "/mnt/beegfs/work/ati/pearl037/regression/mass_range_13.4/random_20sims_200K/averaged_densities" \
              "/log_alpha_" + log_alpha_str + "/"

# Data parameters

path_sims = "/mnt/beegfs/work/ati/pearl037/"
all_sims = ["%i" % i for i in np.arange(22)]
all_sims.remove("3")

path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set_13.4/20sims/random/200k/"

# training

training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

# testing

scaler = load(open(path_data + 'scaler_output.pkl', 'rb'))
large_val_particle_IDs = load(open(path_data + 'larger_validation_set.pkl', 'rb'))
large_val_labels_particle_IDS = load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))

dim = (75, 75, 75)
params_box = {'input_type': 'averaged', 'num_shells': 20}
params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}

# Model params

lr = 0.00005

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

