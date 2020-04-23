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

    # Load data

    path_data = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/"

    scaler_output = load(open(path_data + 'scaler_output.pkl', "rb"))
    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)

    params_val = {'batch_size': 1, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_particle_IDs, s.sims_dic,
                                            shuffle=False, **params_val)

    ######### TRAIN THE MODEL ################

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
                 "/cauchy_selec/"

    trained_model = load_model("/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin"
                               "/larger_net/mse/model/weights.10.hdf5")

    # callbacks
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
    lrate = callbacks.LearningRateScheduler(CNN.lr_scheduler)
    callbacks_list = [checkpoint_call, csv_logger, lrate]

    lr = 0.0001
    Model = CNN.CNN({}, {}, model_type="regression", train=True, compile=True,
                    pretrained_model=trained_model, initial_epoch=10,
                    training_generator=generator_training,
                    validation_generator=generator_validation,
                    lr=0.0001, callbacks=callbacks_list, metrics=['mae', 'mse'],
                    num_epochs=100, dim=generator_validation.dim,
                    loss=lf.cauchy_selection_loss(), validation_steps=len(generator_validation),
                    max_queue_size=10, use_multiprocessing=False, workers=2, verbose=1,
                    num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1)

    # #### RESUME MODEL
    #
    # model = load_model(path_model + "model/weights.20.hdf5",
    #                    custom_objects={'cauchy_selection_loss':lf.cauchy_selection_loss})
    #
    # # callbacks
    # filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    # checkpoint_call = callbacks.ModelCheckpoint(filepath, period=1)
    # csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
    # callbacks_list = [checkpoint_call, csv_logger]
    #
    # history = model.fit_generator(generator=generator_training,
    #                               # validation_data=generator_validation,
    #                               max_queue_size=10, use_multiprocessing=False, workers=1,
    #                               initial_epoch=20,
    #                               verbose=1, epochs=100, shuffle=True,
    #                               callbacks=callbacks_list,
    #                               # validation_freq=5,validation_steps=50
    #                               )
    # pred = model.predict_generator(generator_training, use_multiprocessing=False, workers=1, verbose=1)
    # truth_rescaled = np.array([training_labels_particle_IDS[ID] for ID in training_particle_IDs])
    # h_m_pred = s_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    # true = s_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    #
    # np.save(path_model + "predicted_training_15.npy", h_m_pred)
    # np.save(path_model + "true_training_15.npy", true)
    #
    # # validation set
    #
    # pred = model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
    # truth_rescaled = np.array([val for (key, val) in validation_set.labels_particle_IDS.items()])
    # # h_m_pred = training_set.scaler_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    # # true = training_set.scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    # h_m_pred = s_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    # true = s_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    #
    # np.save(path_model + "predicted_val_15.npy", h_m_pred)
    # np.save(path_model + "true_val_15.npy", true)
