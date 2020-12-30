import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import dump, load
import numpy as np


def predict_from_model(model, epoch, gen_train, gen_val, training_IDs, training_labels_IDS,
                       val_IDs, val_labels_IDs, scaler, saving_path):
    # pred = model.predict_generator(gen_train, use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10)
    # truth_rescaled = np.array([training_labels_IDS[ID] for ID in training_IDs])
    # h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    # true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    # np.save(saving_path + "predicted_training_"+ epoch + ".npy", h_m_pred)
    # np.save(saving_path + "true_training_" + epoch + ".npy", true)
    pred = model.predict_generator(gen_val, use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10)
    truth_rescaled = np.array([val_labels_IDs[ID] for ID in val_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(saving_path + "larger_predicted_val_"+ epoch + ".npy", h_m_pred)
    np.save(saving_path + "larger_true_val_" + epoch + ".npy", true)

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
           "/cauchy_selec_bound_gamma_train_alpha/l2_conv_l21_l1_dense/test/"

    # Convolutional layers parameters

    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    # Dense layers parameters

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    reg_params = {'init_alpha': -3, 'upper_bound_alpha': -3, 'lower_bound_alpha': -4,
                  'init_gamma': 0.2, 'upper_bound_gamma': 0.4, 'lower_bound_gamma': 0.1,
                  'regularizer_conv': reg.l2_norm, 'regularizer_dense': reg.l1_and_l21_group
                  }

    # Regularization parameters + Cauchy likelihood

    path_model = path + "lr/"

    train = False

    # for lr_i in [0.001, 0.01]:
    for lr_i in [0.0005]:
        # p = path_model + str(lr_i) + "/exp_decay/"

        # Train for 100 epochs

        if train:
            Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                                  validation_generator=generator_validation, num_epochs=60, dim=generator_training.dim,
                                  max_queue_size=10, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1,
                                  lr=lr_i, save_summary=True, path_summary=path, validation_freq=1, train=True,
                                  compile=True)

        else:
            weights = path_model + str(lr_i) + "/model/weights.11.h5"

            Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                                  validation_generator=generator_validation, num_epochs=60, dim=generator_training.dim,
                                  max_queue_size=10, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1,
                                  lr=lr_i, save_summary=False, path_summary=path_model + str(lr_i) + "/",
                                  validation_freq=1, train=False, compile=True)

            val_particle_IDs = load(open(path_data + 'larger_validation_set.pkl', 'rb'))
            val_labels_particle_IDS = load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))

            params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
            generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                                    shuffle=False, **params_val)

            generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                                  shuffle=False, **params_tr)

            predict_from_model(Model.model, "11", generator_training, generator_validation,
                               training_particle_IDs, training_labels_particle_IDS,
                               val_particle_IDs, val_labels_particle_IDS, scaler_output, path_model + str(lr_i)+ "/")


