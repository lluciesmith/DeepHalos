import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import dump, load

if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # Load data

    path_data = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/"

    scaler_output = load(open(path_data + 'scaler_output_50000.pkl', "rb"))
    training_particle_IDs = load(open(path_data + 'training_set_50000.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set_50000.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=True, **params_val)


    ######### TRAIN THE MODEL ################

    path = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
           "/cauchy_selec_bound_gamma_train_alpha/"

    # Regularizers are added in `CNNCauchy'

    # conv_l2 = reg.l2_norm(0.1)
    # dense_l21_l1 = reg.l1_and_l21_group(0.1)

    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                       #'kernel_regularizer': conv_l2
                       }
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      #'kernel_regularizer': dense_l21_l1
                      }
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    reg_params = {'init_alpha': 0.1, 'upper_bound_alpha': 0.1, 'lower_bound_alpha': 0.001,
                  'init_gamma': 0.2, 'upper_bound_gamma': 0.4, 'lower_bound_gamma': 0.1
                  }

    # Train for 30 epochs

    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", dim=generator_training.dim,
                          training_generator=generator_training, validation_generator=generator_validation,
                          num_epochs=3, validation_freq=1, lr=0.0001, max_queue_size=10, use_multiprocessing=False,
                          workers=0, verbose=1, num_gpu=1, save_summary=True, path_summary=path,
                          compile=True, train=True,
                          train_mse=False, load_mse_weights=False, load_weights=None, use_tanh_n_epoch=1,
                          **reg_params)

    # Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression",
    #                       training_generator={}, validation_generator={}, dim=(31, 31, 31),
    #                       num_epochs=60, validation_freq=1, lr=0.0001, max_queue_size=10, use_multiprocessing=False,
    #                       workers=0, verbose=1, num_gpu=1, save_summary=True, path_summary=path, compile=True,
    #                       train=False, load_mse_weights=True, **reg_params)

    # alphas = [0.1]
    # gammas = [0.2]
    # for epoch in epochs:
    #     Model.model.load_weights(path + "model/weights." + epoch + ".hdf5")
    #     g = float(K.get_value(Model.model.layers[-1].gamma))
    #     a = float(K.get_value(Model.model.layers[-1].alpha))
    #     gammas.append(g)
    #     alphas.append(a)