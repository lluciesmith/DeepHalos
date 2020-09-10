from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.z0_data_processing as dp0
from pickle import load
import numpy as np


if __name__ == "__main__":

########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    saving_path = "/mnt/beegfs/work/ati/pearl037/regression/mass_range_13.4/random_20sims_200K/z0/"
    seed = 123

    # Load data

    path_sims = "/mnt/beegfs/work/ati/pearl037/"
    all_sims = ["%i" % i for i in np.arange(22)]
    all_sims.remove("3")
    s = dp0.SimulationPreparation_z0(all_sims, path=path_sims)

    path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set_13.4/20sims/random/200k/"
    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    dim = (75, 75, 75)
    params_box = {'res_sim': 1667, 'rescale': True, 'dim': dim}
    generator_training = dp0.DataGenerator_z0(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                              shuffle=True, batch_size=64, **params_box)
    generator_validation = dp0.DataGenerator_z0(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                                shuffle=False, batch_size=50, **params_box)

    ######### TRAIN THE MODEL ################
    lr = 0.00005
    alpha = 10 ** -2.4

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

    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                          shuffle=True, validation_generator=generator_validation, num_epochs=30,
                          metrics=[CNN.likelihood_metric],
                          steps_per_epoch=len(generator_training), validation_steps=len(generator_validation),
                          dim=generator_training.dim, initialiser="Xavier_uniform", max_queue_size=8,
                          use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=lr, save_summary=True,
                          path_summary=saving_path, validation_freq=1, train=True, compile=True,
                          initial_epoch=None, lr_scheduler=False,
                          seed=seed)

