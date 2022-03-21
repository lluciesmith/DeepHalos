from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
import tensorflow as tf
import random as python_random
import sys
import importlib
import training as training_script
import params_ell as params


if __name__ == "__main__":
    idx = int(sys.argv[1])
    scale0 = params.smoothing_scales[idx]
    params.saving_path = params.saving_path + "scale_%.2f/" % float(scale0)

    scaler = load(open(params.saving_path + 'scaler_output_ell.pkl', 'rb'))
    labels_val = training_script.turn_mass_labels_into_ellipticity(params.larger_val_labels_particle_IDS, scale0,
                                                                   params.path_sims, scaler=scaler)
    params.larger_val_labels_particle_IDS = labels_val

    # Load data
    s = tn.SimulationPreparation([params.val_sim], path=params.path_sims)
    generator_validation = tn.DataGenerator(params.larger_val_particle_IDs, params.larger_val_labels_particle_IDS,
                                            s.sims_dic, shuffle=False, **params.params_val)

    # Load the model
    weights = params.saving_path + "model/weights." + params.num_epoch_testing + ".h5"
    Model = CNN.CNNCauchy(params.param_conv, params.param_fcc, model_type="regression", training_generator={}, shuffle=True,
                          validation_generator=generator_validation, num_epochs=100, dim=generator_validation.dim,
                          max_queue_size=80, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=params.lr,
                          save_summary=False, path_summary=params.saving_path, validation_freq=1, train=False, compile=True,
                          initial_epoch=None, initialiser="Xavier_uniform")
    Model.model.load_weights(weights)

    # Predict the labels for the test set

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
    truth_rescaled = np.array([params.larger_val_labels_particle_IDS[ID] for ID in params.larger_val_particle_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(params.saving_path + "predicted_sim_" + params.val_sim + "_epoch_" + params.num_epoch_testing + ".npy",
            h_m_pred)
    np.save(params.saving_path + "true_sim_" + params.val_sim + "_epoch_" + params.num_epoch_testing + ".npy", true)
