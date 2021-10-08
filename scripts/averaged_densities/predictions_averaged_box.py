from dlhalos_code import CNN
import dlhalos_code.data_processing as tn
import numpy as np
import importlib
import sys, os

if __name__ == "__main__":
    try: params = importlib.import_module(sys.argv[1])
    except IndexError: import params_avg as params

    ########### CREATE GENERATORS FOR TESTING #########

    # Load data

    s = tn.SimulationPreparation([params.val_sim], path=params.path_sims)
    generator_validation = tn.DataGenerator(params.large_val_particle_IDs, params.large_val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params.params_val, **params.params_box)


    ######### LOAD THE MODEL ################
    seed = int(sys.argv[2])
    params.saving_path = params.saving_path + "seed_" + str(seed) + "/"
    tr = np.loadtxt(params.saving_path + "training.log", delimiter=",", skiprows=1)
    params.num_epoch_testing = int(np.where(tr[:, 4] == tr[:, 4].min())[0] + 1)

    weights = params.saving_path + "model/weights." + params.num_epoch_testing + ".h5"
    Model = CNN.CNNCauchy(params.param_conv, params.param_fcc, model_type="regression", training_generator={}, shuffle=True,
                          validation_generator=generator_validation, num_epochs=100, dim=generator_validation.dim,
                          max_queue_size=80, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=params.lr,
                          save_summary=False, path_summary=params.saving_path, validation_freq=1, train=False, compile=True,
                          initial_epoch=None, initialiser="Xavier_uniform")
    Model.model.load_weights(weights)

    ######### PREDICT LABELS ################

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
    truth_rescaled = np.array([params.large_val_labels_particle_IDS[ID] for ID in params.large_val_particle_IDs])

    h_m_pred = params.scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = params.scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()

    np.save(params.saving_path + "predicted_sim_" + params.val_sim + "_epoch_" + params.num_epoch_testing + ".npy", h_m_pred)
    np.save(params.saving_path + "true_sim_" + params.val_sim + "_epoch_" + params.num_epoch_testing + ".npy", true)

    if params.test_training is True:
        s = tn.SimulationPreparation(params.all_sims, path=params.path_sims)
        generator_training = tn.DataGenerator(params.training_particle_IDs, params.training_labels_particle_IDS, s.sims_dic,
                                              shuffle=True, **params.params_tr, **params.params_box)

        predt = Model.model.predict_generator(generator_training, use_multiprocessing=False, workers=0, verbose=1)
        truth_rescaledt = np.array([params.training_labels_particle_IDS[ID] for ID in params.training_particle_IDs])

        h_m_predt = params.scaler.inverse_transform(predt.reshape(-1, 1)).flatten()
        truet = params.scaler.inverse_transform(truth_rescaledt.reshape(-1, 1)).flatten()

        np.save(params.saving_path + "training_predicted_epoch_" + params.num_epoch + ".npy", h_m_predt)
        np.save(params.saving_path + "training_true_sim_epoch_" + params.num_epoch + ".npy", truet)
