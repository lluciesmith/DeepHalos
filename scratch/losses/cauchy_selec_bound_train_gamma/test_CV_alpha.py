import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from utilss import kl_divergence as kl
from pickle import dump, load
import numpy as np
import os


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # Load data

    path_data = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/"

    scaler_output = load(open(path_data + 'scaler_output_50000.pkl', "rb"))
    val_particle_IDs = load(open(path_data + 'larger_validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    all_sims = ["6"]
    s = tn.SimulationPreparation(all_sims)

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)


    ######### TEST THE MODEL ################

    path = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
           "/cauchy_selec_bound_gamma_cv_alpha/"

    # Define model

    s_weights = []
    neurons_active = []
    KL_divergence = []

    alpha_grid = [10**-j for j in np.arange(1, 5).astype("float")]
    for alpha in alpha_grid:
        path_model = path + "alpha_" + str(alpha) + "/"

        # Here we do not train alpha but we do a grid search

        conv_l2 = reg.l2_norm(alpha)
        dense_l21_l1 = reg.l1_and_l21_group(alpha)

        params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                           'kernel_regularizer': conv_l2}
        param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                      'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                      'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                      'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                      'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                      }

        params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True, 'kernel_regularizer': dense_l21_l1}
        param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc}, 'last': {}}

        # Compile model using weights where validation loss is at its minimum

        loss = np.loadtxt(path_model + "training.log", delimiter=",", skiprows=1)
        min_epoch = '%02i' % (loss[np.where(loss[:, 2] == loss[:, 2].min())[0], 0] + 1)
        print("Epoch for " + str(alpha) + " model is " + min_epoch)
        best_weights = path_model + "model/weights." + min_epoch + ".hdf5"

        Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", dim=generator_validation.dim,
                              path_summary=path_model, train=False, compile=True)
        model = Model.model

        # Get predictions validation set

        pred = model.predict_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
        truth_rescaled = np.array([val_labels_particle_IDS[ID] for ID in val_particle_IDs])
        h_m_pred = scaler_output.inverse_transform(pred.reshape(-1, 1)).flatten()
        true = scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
        np.save(path_model + "predicted_val_" + min_epoch + ".npy", h_m_pred)
        np.save(path_model + "true_val_" + min_epoch + ".npy", true)

        # Get KL divergence

        KL = kl.get_KL_div(true, h_m_pred, bandwidth=0.1)
        KL_divergence.append(KL)

        # Get sparsity

        sparse_alpha = reg.active_neurons(model)
        neurons_active.append(sparse_alpha)

        # Get active neurons
        n = reg.sparsity_weights(model)
        s_weights.append(n)

    np.save(path + "sparsity_weights.npy", s_weights)
    np.save(path + "active_neurons.npy", neurons_active)
    np.save(path + "KL_divergences.npy", KL_divergence)