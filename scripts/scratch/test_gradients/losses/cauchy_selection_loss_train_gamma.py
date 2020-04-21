import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras as keras
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

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net" \
                 "/cauchy_selec/train_gamma/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    # Use the same training set/validation set as for cauchy+selection loss function

    params_inputs = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    tr_set = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/cauchy_selec/"

    training_particle_IDs = load(open(tr_set + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(tr_set + 'labels_training_set.pkl', 'rb'))
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=False, **params_inputs)

    # validation set

    validation_particle_IDs = load(open(tr_set + 'validation_set.pkl', 'rb'))
    validation_labels_particle_IDS = load(open(tr_set + 'labels_validation_set.pkl', 'rb'))
    generator_validation = tn.DataGenerator(validation_particle_IDs, validation_labels_particle_IDS, s.sims_dic,
                                            **params_inputs)

    ######### TRAINING MODEL FROM MSE TRAINED ONE ##############

    lr = 0.0001

    model_mse = load_model(path_transfer + "model/weights.10.hdf5")
    predictions = CNN.CauchyLayer()(model_mse.layers[-1].output)
    trained_model = keras.Model(inputs=model_mse.input, outputs=predictions)

    optimiser = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
    loss_c = lf.cauchy_selection_loss_trainable_gamma(trained_model.layers[-1])
    trained_model.compile(loss=loss_c, optimizer=optimiser, metrics=['mae', 'mse'])
    trained_model.save_weights(path_model + 'model/initial_weights.h5')

    # callbacks
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
    callbacks_list = [checkpoint_call, csv_logger]

    history = trained_model.fit_generator(generator=generator_training,
                                          # validation_data=generator_validation,
                                          use_multiprocessing=False, workers=1, max_queue_size=10, verbose=1,
                                          epochs=100, shuffle=True, callbacks=callbacks_list, initial_epoch=10)

    #### RESUME MODEL

    # model = load_model(path_model + "model/weights.20.hdf5",
    #                    custom_objects={'cauchy_selection_loss':lf.cauchy_selection_loss})

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
