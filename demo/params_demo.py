from dlhalos_code import custom_regularizers as reg

saving_path = "./demo-data/"
path_sims = "./demo-data/"
path_data = "./demo-data/"
seed = 123

# Data parameters

dim = (11, 11, 11)
params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}

# Model parameters

lr = 0.0001
alpha = 10**-2.5

params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                   'kernel_regularizer': reg.l2_norm(alpha)}
param_conv = {'conv_1': {'num_kernels': 8, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
              'conv_2': {'num_kernels': 8, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_3': {'num_kernels': 16, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
              }

params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                  'kernel_regularizer': reg.l1_and_l21_group(alpha)}
param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
             'last': {}}




