import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
import tensorflow
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
from pickle import dump, load
import tensorflow.keras.backend as K


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/custom/"
    path_training_set = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/mse2/"

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
        training_particle_IDs = load(open(path_training_set + 'training_set.pkl', 'rb'))
        training_labels_particle_IDS = load(open(path_training_set + 'labels_training_set.pkl', 'rb'))
        s_output = load(open(path_training_set + 'scaler_output.pkl', "rb"))
        print("loaded training set")
    except OSError:
        training_set = tn.InputsPreparation(train_sims, scaler_type="minmax",
                                            load_ids=False, shuffle=True, log_high_mass_limit=13,
                                            random_style="uniform", random_subset_each_sim=1000000,
                                            num_per_mass_bin=10000)
        dump(training_set.particle_IDs, open(path_model + 'training_set.pkl', 'wb'))
        dump(training_set.labels_particle_IDS, open(path_model + 'labels_training_set.pkl', 'wb'))
        dump(training_set.scaler_output, open(path_model + 'scaler_output.pkl', 'wb'))
        training_particle_IDs = training_set.particle_IDs
        training_labels_particle_IDS = training_set.labels_particle_IDS
        s_output = training_set.scaler_output

    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_inputs)

    # validation set

    validation_set = tn.InputsPreparation([val_sim], load_ids=False, random_subset_each_sim=100000,
                                          num_per_mass_bin=100, log_high_mass_limit=13, scaler_output=s_output)
    generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
                                            **params_inputs)

    ######### TRAINING MODEL ##############

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

    # save histories
    csv_logger = CSVLogger(path_model + "/training.log", separator=',')

    callbacks_list = [checkpoint_call, csv_logger]

    tensorflow.compat.v1.set_random_seed(7)

    kernel_reg = regularizers.l2(0.00001)
    bias_reg = regularizers.l2(0.00001)
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
                  'conv_4': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), **params_all_conv},
                  # 'conv_5': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), **params_all_conv}
                  }

    params_all_fcc = {'bn': False, 'dropout': 0.4, 'activation': activation, 'relu': relu}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc},
                 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}
                 }

    def custom_loss(y_true, y_predicted):
        epsilon = 10**-10
        r = y_true - y_predicted
        factor = K.log((1 - K.exp((-r**2 + epsilon) / 2))/(r**2+ epsilon))
        # norm = K.log(2)
        return - K.mean(factor, axis=-1)

    lr = 0.0001
    Model = CNN.CNN(param_conv, param_fcc, model_type="regression",
                    training_generator=generator_training, validation_generator=generator_validation,
                    lr=lr, callbacks=callbacks_list, metrics=['mae', 'mse'],
                    num_epochs=100, dim=params_inputs['dim'], loss=custom_loss,
                    max_queue_size=10, use_multiprocessing=True, workers=2, verbose=1,
                    num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1, train=True,
                    compile=True)

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


# model = load_model(path_model + "model/weights.50.hdf5", custom_objects={'cauchy_loss': cauchy_loss})
# #model= load_model(path_model + "model/weights.10.hdf5")
#
# # # training set
# #
# pred = model.predict_generator(generator_training, use_multiprocessing=False, workers=1, verbose=1)
# truth_rescaled = np.array([training_labels_particle_IDS[ID] for ID in training_particle_IDs])
# h_m_pred = s_output.inverse_transform(pred.reshape(-1, 1)).flatten()
# true = s_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
#
# np.save(path_model + "predicted_training_50.npy", h_m_pred)
# np.save(path_model + "true_training_50.npy", true)
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
# np.save(path_model + "predicted_val_50.npy", h_m_pred)
# np.save(path_model + "true_val_50.npy", true)


# class MyLayer(Layer):
#
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(MyLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         init = keras.initializers.RandomNormal(mean=0.0, stddev=1)
#         self.gamma = self.add_weight(name='gamma',
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer=init,
#                                       trainable=True)
#         super(MyLayer, self).build(input_shape)  # Be sure to call this at the end
#
#     def call(self, x):
#         return x
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)
#
#
# def custom_loss(layer):
#     # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
#     def loss(y_true, y_pred):
#         return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)
#
#     # Return a function
#     return loss
#
#
# M = MyLayer(1)
# g = M.gamma
#
# def cauchy_loss(layer):
#     # w = MyLayer().gamma
#     # # init = keras.initializers.RandomNormal(mean=0.0, stddev=1)
#     # # w = K.variable(init(shape=(1,)), dtype='float32', name="gamma")
#     def loss(y_true, y_pred):
#         logl = K.log(K.square(y_true - y_pred) + K.square(layer.kernel))
#         return K.mean(logl, axis=-1)
#     return loss