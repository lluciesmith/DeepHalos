import dlhalos_code.z0_data_processing as dp0
from dlhalos_code import CNN
import numpy as np
import importlib
import sys

if __name__ == "__main__":
    try: params = importlib.import_module(sys.argv[1])
    except IndexError: import params_z0 as params

    ########### CREATE GENERATORS FOR VALIDATION #########

    s = dp0.SimulationPreparation_z0([params.val_sim], path=params.path_sims)
    generator_validation = dp0.DataGenerator_z0(params.large_val_particle_IDs, params.large_val_labels_particle_IDS,
                                                s.sims_dic, shuffle=False, batch_size=50,
                                                **params.params_box)

    ######### LOAD THE MODEL ################

    weights = params.saving_path + "model/weights." + params.num_epoch + ".h5"
    Model = CNN.CNNCauchy(params.param_conv, params.param_fcc, model_type="regression", training_generator={},
                          shuffle=True, validation_generator=generator_validation, num_epochs=100,
                          dim=generator_validation.dim, max_queue_size=80, use_multiprocessing=False, workers=0,
                          verbose=1, num_gpu=1, lr=params.lr, save_summary=False, path_summary=params.saving_path,
                          validation_freq=1, train=False, compile=True, initial_epoch=None,
                          initialiser="Xavier_uniform")
    Model.model.load_weights(weights)

    ######### PREDICT LABELS ################

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
    truth_rescaled = np.array([params.large_val_labels_particle_IDS[ID] for ID in params.large_val_particle_IDs])
    h_m_pred = params.scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = params.scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()

    np.save(params.saving_path + "predicted_sim_" + params.val_sim + "_epoch_" + params.num_epoch_testing + ".npy",
            h_m_pred)
    np.save(params.saving_path + "true_sim_" + params.val_sim + "_epoch_" + params.num_epoch_testing + ".npy", true)
