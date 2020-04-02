import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
import tensorflow
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
import numpy as np
from pickle import dump, load
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error as mse


if __name__ == "__main__":

########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/custom_loss/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    params_inputs = {'batch_size': 100,
                     'rescale_mean': 1.005,
                     'rescale_std': 0.05050,
                     'dim': (31, 31, 31)
                     }

    # define a common scaler for the output

    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    training_set = tn.InputsPreparation(train_sims, load_ids=False, shuffle=True, scaler_type="minmax",
                                        log_high_mass_limit=13,
                                        random_style="uniform", random_subset_each_sim=1000000, num_per_mass_bin=1000,
                                        # random_subset_each_sim=1000
                                        )
    generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS,
                                              s.sims_dic, **params_inputs)

    validation_set = tn.InputsPreparation([val_sim], load_ids=False, random_subset_each_sim=10000,
                                          log_high_mass_limit=13,
                                          scaler_output=training_set.scaler_output, shuffle=True)
    generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS,
                                              s.sims_dic, **params_inputs)
    dump(training_set.scaler_output, open(path_model + 'scaler_output.pkl', 'wb'))

    ######### TRAINING MODEL ##############

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

    # save histories
    csv_logger = CSVLogger(path_model + "/training.log", separator=',')

    # tensorboard
    # tb = TensorBoard(log_dir=path_model + '/logs', histogram_freq=1, update_freq='epoch',
    #                  write_grads=True, write_graph=False)

    # learning rate scheduler
    # lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    callbacks_list = [checkpoint_call, csv_logger]

    tensorflow.compat.v1.set_random_seed(7)

    kernel_reg = regularizers.l2(0.0005)
    bias_reg = regularizers.l2(0.0005)
    activation = "linear"
    relu = True

    params_all_conv = {'activation': activation, 'relu': relu,
                       'strides': 1, 'padding': 'same', 'pool': "max", 'bn': False,
                       'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg
                       }
    param_conv = {'conv_1': {'num_kernels': 8, 'dim_kernel': (5, 5, 5), **params_all_conv},
                  'conv_2': {'num_kernels': 16, 'dim_kernel': (3, 3, 3), **params_all_conv},
                  'conv_3': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), **params_all_conv},
                  }

    params_all_fcc = {'bn': False,
                      'dropout': 0.4,
                      'activation': activation, 'relu': relu,
                      'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg
                      }
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc},
                 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {
                     # 'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg
                 }
                 }

    def custom_loss(y_true, y_predicted):
        factor = (1 + abs(y_true)) ** 6
        return K.mean(factor * K.square(y_predicted - y_true), axis=-1)

    loss = custom_loss
    Model = CNN.CNN(param_conv, param_fcc, model_type="regression",
                    training_generator=generator_training, validation_generator=generator_validation,
                    lr=0.001, callbacks=callbacks_list, metrics=['mae', 'mse'],
                    num_epochs=100, dim=params_inputs['dim'], loss=loss,
                    max_queue_size=10, use_multiprocessing=True, workers=2, verbose=1,
                    num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1, train=True)


    ############# PREDICTIONS #############

    # training set

    pred = Model.model.predict_generator(generator_training, use_multiprocessing=False, workers=1, verbose=1)
    truth_rescaled = np.array([val for (key, val) in training_set.labels_particle_IDS.items()])
    h_m_pred = training_set.scaler_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = training_set.scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()

    np.save(path_model + "predicted_training.npy", h_m_pred)
    np.save(path_model + "true_training.npy", true)

    # validation set

    pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
    truth_rescaled = np.array([val for (key, val) in validation_set.labels_particle_IDS.items()])
    h_m_pred = training_set.scaler_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = training_set.scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()

    np.save(path_model + "predicted_val.npy", h_m_pred)
    np.save(path_model + "true_val.npy", true)


    # model= load_model(path_model + "model/weights.10.hdf5")
    #
    # training set
    #
    # pred = model.predict_generator(generator_training, use_multiprocessing=False, workers=1, verbose=1)
    # truth_rescaled = np.array([val for (key, val) in training_set.labels_particle_IDS.items()])
    # h_m_pred = training_set.scaler_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    # true = training_set.scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    #
    # np.save(path_model + "predicted_training_45.npy", h_m_pred)
    # np.save(path_model + "true_training_45.npy", true)
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
    # np.save(path_model + "predicted_val_45.npy", h_m_pred)
    # np.save(path_model + "true_val_45.npy", true)
