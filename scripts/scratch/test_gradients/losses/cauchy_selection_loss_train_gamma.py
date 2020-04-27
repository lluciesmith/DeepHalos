import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
from pickle import dump, load
import tensorflow.keras as keras
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

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)


    ######### TRAINING MODEL FROM MSE TRAINED ONE ##############

    # Load weights and model from MSE run

    trained_weights = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin" \
                      "/larger_net/lr_decay/mse/model/weights.10.hdf5"

    kernel_reg = regularizers.l2(0.0005)
    bias_reg = regularizers.l2(0.0005)
    activation = "linear"
    relu = True

    params_all_conv = {'activation': activation, 'relu': relu, 'strides': 1, 'padding': 'same',
                       'bn': False, 'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    params_all_fcc = {'bn': False, 'dropout': 0.4, 'activation': activation, 'relu': relu,
                      'kernel_regularizer': kernel_reg}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc},
                 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}
                 }

    params_all_conv = {'activation': activation, 'relu': relu, 'strides': 1, 'padding': 'same',
                       'bn': False, 'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    params_all_fcc = {'bn': False, 'dropout': 0.4, 'activation': activation, 'relu': relu,
                      'kernel_regularizer': kernel_reg}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc},
                 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}
                 }

    M = CNN.CNN(param_conv, param_fcc, model_type="regression", train=True, compile=True, weights=trained_weights,
                initial_epoch=10,  training_generator=generator_training, validation_generator=generator_validation,
                lr=0.0001,
                # callbacks=callbacks_list,
                # metrics=['mae', 'mse'],
                num_epochs=100, dim=generator_validation.dim,
                loss=lf.cauchy_selection_loss_fixed_boundary(), validation_steps=len(generator_validation),
                max_queue_size=10, use_multiprocessing=False, workers=2, verbose=1,
                num_gpu=1, save_summary=False, validation_freq=1)

    # Compute new model

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
                 "/cauchy_selec_gamma/"

    model_mse = M.model
    predictions = CNN.CauchyLayer()(model_mse.layers[-1].output)
    trained_model = keras.Model(inputs=model_mse.input, outputs=predictions)

    optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
    loss_c = lf.cauchy_selection_loss_trainable_gamma(trained_model.layers[-1])
    trained_model.compile(loss=loss_c, optimizer=optimiser, metrics=['mae', 'mse'])
    trained_model.save_weights(path_model + 'model/initial_weights.h5')

    # callbacks
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5, save_weights_only=True)
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
    lrate = callbacks.LearningRateScheduler(CNN.lr_scheduler)
    callbacks_list = [checkpoint_call, csv_logger, lrate]

    history = trained_model.fit_generator(generator=generator_training,
                                          # validation_data=generator_validation,
                                          # validation_steps=len(generator_validation),
                                          use_multiprocessing=False, workers=1, max_queue_size=10, verbose=1,
                                          epochs=100, shuffle=True, callbacks=callbacks_list, initial_epoch=10)

    #### RESUME MODEL

    # x = tf.constant(value=3)
    # L = CNN.CauchyLayer()
    # y = L(x)
    #
    # model = load_model(path_model + "model/weights.20.hdf5",
    #                    custom_objects={'loss':lf.cauchy_selection_loss_trainable_gamma(CNN.CauchyLayer),
    #                                    'CauchyLayer': L})

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
