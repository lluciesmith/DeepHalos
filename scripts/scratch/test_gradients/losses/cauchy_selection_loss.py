import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
from pickle import dump, load
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import load_model


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    path_transfer = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/mse/"
    scaler_training_set = load(open(path_transfer + 'scaler_output.pkl', 'rb'))

    # Create new model

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/cauchy_selec/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    params_inputs = {'batch_size': 100,
                     'rescale_mean': 1.005,
                     'rescale_std': 0.05050,
                     'dim': (31, 31, 31),
                     # 'shuffle': True
                     }
    try:
        training_particle_IDs = load(open(path_model + 'training_set.pkl', 'rb'))
        training_labels_particle_IDS = load(open(path_model + 'labels_training_set.pkl', 'rb'))
        s_output = load(open(path_model + 'scaler_output.pkl', "rb"))
        print("loaded training set")
    except OSError:
        training_set = tn.InputsPreparation(train_sims, scaler_type="minmax", output_range=(-1, 1),
                                            load_ids=False, shuffle=True,
                                            log_high_mass_limit=13,
                                            random_style="uniform", random_subset_each_sim=1000000,
                                            num_per_mass_bin=1000, scaler_output=scaler_training_set)
        dump(training_set.particle_IDs, open(path_model + 'training_set.pkl', 'wb'))
        dump(training_set.labels_particle_IDS, open(path_model + 'labels_training_set.pkl', 'wb'))
        dump(training_set.scaler_output, open(path_model + 'scaler_output.pkl', 'wb'))
        training_particle_IDs = training_set.particle_IDs
        training_labels_particle_IDS = training_set.labels_particle_IDS
        s_output = training_set.scaler_output

    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=False, **params_inputs)

    # validation set
    #
    # validation_set = tn.InputsPreparation([val_sim], load_ids=False, random_subset_each_sim=10000,
    #                                       log_high_mass_limit=13, scaler_output=scaler_training_set)
    # generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
    #                                         **params_inputs)

    ######### TRAINING MODEL FROM MSE TRAINED ONE ##############

    trained_model = load_model(path_transfer + "model/weights.10.hdf5")

    # callbacks
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
    callbacks_list = [checkpoint_call, csv_logger]

    lr = 0.0001
    Model = CNN.CNN({}, {}, model_type="regression", train=True, compile=True,
                    pretrained_model=trained_model, initial_epoch=10,
                    training_generator=generator_training,
                    # validation_generator=generator_validation,
                    lr=0.0001, callbacks=callbacks_list, metrics=['mae', 'mse'],
                    num_epochs=100, dim=params_inputs['dim'],
                    loss=lf.cauchy_selection_loss, validation_steps=50,
                    max_queue_size=10, use_multiprocessing=False, workers=1, verbose=1,
                    num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1)

    #### RESUME MODEL

    model = load_model(path_model + "model/weights.20.hdf5",
                       custom_objects={'cauchy_selection_loss':lf.cauchy_selection_loss})

    # callbacks
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=1)
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
    callbacks_list = [checkpoint_call, csv_logger]

    history = model.fit_generator(generator=generator_training,
                                  # validation_data=generator_validation,
                                  max_queue_size=10, use_multiprocessing=False, workers=1,
                                  initial_epoch=20,
                                  verbose=1, epochs=100, shuffle=True,
                                  callbacks=callbacks_list,
                                  # validation_freq=5,validation_steps=50
                                  )
    pred = model.predict_generator(generator_training, use_multiprocessing=False, workers=1, verbose=1)
    truth_rescaled = np.array([training_labels_particle_IDS[ID] for ID in training_particle_IDs])
    h_m_pred = s_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = s_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()

    np.save(path_model + "predicted_training_15.npy", h_m_pred)
    np.save(path_model + "true_training_15.npy", true)

    # validation set

    pred = model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
    truth_rescaled = np.array([val for (key, val) in validation_set.labels_particle_IDS.items()])
    # h_m_pred = training_set.scaler_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    # true = training_set.scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    h_m_pred = s_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = s_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()

    np.save(path_model + "predicted_val_15.npy", h_m_pred)
    np.save(path_model + "true_val_15.npy", true)
