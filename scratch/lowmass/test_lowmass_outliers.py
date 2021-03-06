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
from collections import OrderedDict
from sklearn.metrics import mean_squared_error as mse

# class MyCustomCallback(callbacks.Callback):
#
#   def on_epoch_end(self, epoch, logs='logs_samples.log'):
#       l = self.model.evaluate()
#       err =
#
#
# def custom_loss():
#     dof = K.variable(1, dtype='float64', name='dof')


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/mse2/"

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

    # training set

    try:
        training_particle_IDs = load(open(path_model + 'training_set.pkl', 'rb'))
        training_labels_particle_IDS = load(open(path_model + 'labels_training_set.pkl', 'rb'))
        s_output = load(open(path_model + 'scaler_output.pkl', "rb"))
        print("loaded training set")
    except OSError:
        training_set = tn.InputsPreparation(train_sims, scaler_type="minmax",
                                            load_ids=False, shuffle=True, log_high_mass_limit=13,
                                            random_style="uniform", random_subset_each_sim=1000000, num_per_mass_bin=10000)
        dump(training_set.particle_IDs, open(path_model + 'training_set.pkl', 'wb'))
        dump(training_set.labels_particle_IDS, open(path_model + 'labels_training_set.pkl', 'wb'))
        dump(training_set.scaler_output, open(path_model + 'scaler_output.pkl', 'wb'))
        training_particle_IDs = training_set.particle_IDs
        training_labels_particle_IDS = training_set.labels_particle_IDS
        s_output = training_set.scaler_output

    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic, **params_inputs)

    # validation set

    validation_set = tn.InputsPreparation([val_sim], load_ids=False, random_subset_each_sim=100000, num_per_mass_bin=100,
                                          log_high_mass_limit=13, scaler_output=s_output, shuffle=True)
    generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
                                            **params_inputs)

    ######### TRAINING MODEL ##############

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=1)

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
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'activation': activation, 'relu': relu,
                             'strides': 1, 'padding': 'same', 'pool': None, 'bn': False,
                             'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), **params_all_conv},
                  }

    params_all_fcc = {'bn': False, 'dropout': 0.4, 'activation': activation, 'relu': relu}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc},
                 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}
                 }

    loss = 'mse'
    Model = CNN.CNN(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                    validation_generator=generator_validation, callbacks=callbacks_list, metrics=['mae', 'mse'],
                    num_epochs=50, dim=params_inputs['dim'], max_queue_size=10, use_multiprocessing=True, workers=2,
                    verbose=1, num_gpu=1, lr=0.001, loss=loss, save_summary=True, path_summary=path_model,
                    validation_freq=1, train=True)

    # model = Model.model
    # epochs = Model.num_epochs
    #
    # dict_err = OrderedDict([(key, []) for key in training_particle_IDs])
    # y_truth = np.array([val for (key, val) in training_labels_particle_IDS.items()])
    #
    # for i, epoch in enumerate(epochs):
    #     history = model.fit_generator(generator=generator_training,
    #                                   # validation_data=generator_validation,
    #                                   use_multiprocessing=False, workers=1, max_queue_size=10, verbose=1,
    #                                   epochs=epoch + 1, shuffle=True, callbacks=callbacks_list, validation_freq=1,
    #                                   initial_epoch=epoch)
    #     model_epoch = load_model(params_model['path_model'] + "model/weights." + str(epoch) + ".hdf5")
    #     pred_tr = model.predict_generator(generator_training, use_multiprocessing=False, workers=1, max_queue_size=1)
    #     err = pred_tr - y_truth
    #     for i, key in enumerate(generator_training.particle_IDs):
    #         dict_err[key].append(err[i])


    ############# PREDICTIONS #############

    # training set

    # pred = Model.model.predict_generator(generator_training, use_multiprocessing=False, workers=1, verbose=1)
    # truth_rescaled = np.array([val for (key, val) in training_set.labels_particle_IDS.items()])
    # h_m_pred = training_set.scaler_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    # true = training_set.scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    #
    # np.save(path_model + "predicted_training.npy", h_m_pred)
    # np.save(path_model + "true_training.npy", true)
    #
    # # validation set
    #
    # pred = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
    # truth_rescaled = np.array([val for (key, val) in validation_set.labels_particle_IDS.items()])
    # h_m_pred = training_set.scaler_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    # true = training_set.scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    #
    # np.save(path_model + "predicted_val.npy", h_m_pred)
    # np.save(path_model + "true_val.npy", true)


    # model = load_model(path_model + "model/weights.10.hdf5", custom_objects={'custom_loss': custom_loss})
    # #model= load_model(path_model + "model/weights.10.hdf5")
    #
    # # # training set
    # #
    # pred = model.predict_generator(generator_training, use_multiprocessing=False, workers=1, verbose=1)
    # truth_rescaled = np.array([val for (key, val) in training_set.labels_particle_IDS.items()])
    # h_m_pred = s_output.inverse_transform(pred.reshape(-1, 1)).flatten()
    # true = s_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    #
    # np.save(path_model + "predicted_training_30.npy", h_m_pred)
    # np.save(path_model + "true_training_30.npy", true)
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
    # np.save(path_model + "predicted_val_30.npy", h_m_pred)
    # np.save(path_model + "true_val_30.npy", true)