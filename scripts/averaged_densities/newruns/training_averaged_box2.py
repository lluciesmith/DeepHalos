from dlhalos_code import CNN
import dlhalos_code.data_processing as tn
import numpy as np

if __name__ == "__main__":
    import params_avg as params

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    s = tn.SimulationPreparation(params.all_sims, path=params.path_sims)

    # Create the generators for training

    generator_training = tn.DataGenerator(params.training_particle_IDs, params.training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, path=params.path_data, **params.params_tr, **params.params_box)
    generator_validation = tn.DataGenerator(params.val_particle_IDs, params.val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, path=params.path_data, **params.params_val, **params.params_box)

    ######### TRAIN THE MODEL ################
    params.lr = 0.0001
    params.saving_path = params.saving_path + "/lr_" + str(params.lr) + "/"
    Model = CNN.CNNGaussian(params.param_conv, params.param_fcc,
                            training_generator=generator_training, steps_per_epoch=int(len(generator_training)/4),
                            validation_generator=generator_validation, num_epochs=20, dim=generator_training.dim,
                            initialiser="Xavier_uniform", max_queue_size=8, use_multiprocessing=False, workers=1,
                            verbose=1, num_gpu=1, lr=params.lr, save_summary=True, path_summary=params.saving_path,
                            train=True, compile=True, initial_epoch=0, lr_scheduler=False, seed=params.seed)

    # test it on the validation set

    tr = np.loadtxt(params.saving_path + "training.log", delimiter=",", skiprows=1)
    num_epoch_testing = str(int(np.where(tr[:, 2] == tr[:, 2].min())[0] + 1))
    weights = params.saving_path + "model/weights." + num_epoch_testing + ".h5"
    Model.model.load_weights(weights)

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1)
    truth_rescaled = np.array([params.val_labels_particle_IDS[ID] for ID in params.val_particle_IDs])
    h_m_pred = params.scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = params.scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(params.saving_path + "predicted_sim_valset_epoch_" + num_epoch_testing + ".npy", h_m_pred)
    np.save(params.saving_path + "true_sim_valset_epoch_" + num_epoch_testing + ".npy", true)
