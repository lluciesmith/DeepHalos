import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims, path="/mnt/beegfs/work/ati/pearl037/")

    # Load data

    path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set/random/"
    num_sample = 50000
    saving_path = path_data + str(num_sample) + "/"

    scaler_output = load(open(saving_path + 'scaler_output.pkl', 'rb'))
    training_particle_IDs = load(open(saving_path + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(saving_path + 'labels_training_set.pkl', 'rb'))

    val_particle_IDs = load(open(saving_path + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(saving_path + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    dim = (75, 75, 75)
    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_tr)


    ######### TRAIN THE MODEL ################

    alpha = 10**-4
    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                       'kernel_regularizer': reg.l2_norm(alpha)}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  }
    # Added conv_6 in going from 31^3 input to 75^3 input

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l1_and_l21_group(alpha)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    reg_params = {'init_gamma': 0.2}

    # Train for one epoch using MSE loss

    lr_i = 5*10**-4
    # lr_i = 10**-5
    path_model = "/mnt/beegfs/work/ati/pearl037/regression/run_rand_tr_" + str(num_sample) + "/"
    # path_model = "/mnt/beegfs/work/ati/pearl037/regression/"
    # weigths = path_model + "model/weights.05.h5"

    Model = CNN.CNNCauchy(param_conv, param_fcc, lr=lr_i, lr_scheduler=False, model_type="regression",
                      dim=generator_training.dim, training_generator=generator_training,
                      validation_generator=generator_validation, num_epochs=60, validation_freq=1,
                      max_queue_size=10, use_multiprocessing=False,  workers=0, verbose=1, num_gpu=1,
                      save_summary=True, path_summary=path_model,
                      compile=True, train=True, load_weights=None,
                      load_mse_weights=False, use_mse_n_epoch=1, use_tanh_n_epoch=0, **reg_params)


