import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv3D, Flatten
from tensorflow import set_random_seed
import sklearn.preprocessing
#
#
# class CNN:
#     def __init__(self, input_shape_box=(17, 17, 17, 1), data_format="channels_last", num_convolutions=1,
#                    num_kernels=3, dim_kernel=(7, 7, 7), strides=2,  padding='valid',
#                    alpha_relu=0.3, activation=True, bn=True, pool=True, dense_neurons=8):


def normalise_output(output, take_log=True):
    if take_log is True:
        log_output = np.log10(output[output > 0])
    else:
        log_output = output
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    minmax_scaler.fit(log_output.reshape(-1, 1))

    normalised_labels = np.zeros((len(output),))
    normalised_labels[output > 0] = minmax_scaler.transform(log_output.reshape(-1, 1)).flatten()
    return minmax_scaler, normalised_labels


def first_convolutional_layer(input_data, input_shape_box=(17, 17, 17, 1),
                              num_kernels=3, dim_kernel=(7, 7, 7), strides=2,  padding='valid',
                              data_format="channels_last", alpha_relu=0.3,
                              activation=True, bn=True, pool=True):

    x = keras.layers.Conv3D(num_kernels, dim_kernel, strides=strides, padding=padding, data_format=data_format,
                            input_shape=input_shape_box)(input_data)
    if activation is True:
        x = keras.layers.LeakyReLU(alpha=alpha_relu)(x)
    if bn is True:
        x = keras.layers.BatchNormalization(axis=-1)(x)
    if pool is True:
        x = keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding=padding, data_format=data_format)(x)
    return x


def subsequent_convolutional_layer(x, num_kernels=3, dim_kernel=(7, 7, 7), strides=2, padding='valid',
                                   data_format="channels_last", alpha_relu=0.3,
                                   activation=True, bn=True, pool=True):
    x = keras.layers.Conv3D(num_kernels, dim_kernel, strides=strides, padding=padding, data_format=data_format)(x)
    if activation is True:
        x = keras.layers.LeakyReLU(alpha=alpha_relu)(x)
    if bn is True:
        x = keras.layers.BatchNormalization(axis=-1)(x)
    if pool is True:
        x = keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding=padding, data_format=data_format)(
            x)
    return x


def model_w_layers(input_shape_box, conv_params, fcc_params, data_format="channels_last"):

    input_data = Input(shape=(*input_shape_box, 1))
    num_convolutions = len(conv_params)
    num_fully_connected = len(fcc_params)

    x = first_convolutional_layer(input_data, input_shape_box=input_shape_box, **conv_params['conv_1'])

    if num_convolutions > 1:
        for i in range(1, num_convolutions):
            params = conv_params['conv_' + str(i + 1)]
            x = subsequent_convolutional_layer(x, **params)

    # Flatten and fully connected layers, followed by dropout

    x = Flatten(data_format=data_format)(x)

    if num_fully_connected > 1:
        for i in range(num_fully_connected):
            params = fcc_params['dense_' + str(i + 1)]
            x = Dense(params['neurons'], activation='relu')(x)
            x = keras.layers.Dropout(params['dropout'])(x)

    predictions = Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input_data, outputs=predictions)
    model.compile(optimizer='adam', loss='mse', metrics=["mae"])
    return model

def fit_model(model, epochs=5, batch_size=32):
    history = model.fit(a, b, batch_size=batch_s, verbose=1, epochs=num_epochs,
                        # validation_split=0.2
                        )



if __name__ == "__main__":

    # Load inputs

    p_ids = np.load("/Users/lls/Documents/deep_halos_files/particles.npy")
    p_inputs = np.load("/Users/lls/Documents/deep_halos_files/3d_inputs_particles.npy")
    m = np.load("/Users/lls/Documents/deep_halos_files/outputs_particles.npy")

    param_conv = {'conv_1': {'num_kernels': 2, 'dim_kernel': (48, 48, 48), 'strides': 1, 'padding':'valid',
                             'pool': True, 'bn': True},
                  'conv_2': {'num_kernels': 12, 'dim_kernel': (22, 22, 22), 'strides': 1, 'padding':'valid',
                             'pool': True, 'bn': True},
                  'conv_3': {'num_kernels': 32, 'dim_kernel': (9, 9, 9), 'strides': 1, 'padding':'valid',
                             'pool': False, 'bn': True},
                  'conv_4': {'num_kernels': 64, 'dim_kernel': (6, 6, 6), 'strides': 1, 'padding':'valid',
                             'pool': False, 'bn': True},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (4, 4, 4), 'strides': 1, 'padding':'valid',
                             'pool': False, 'bn': True},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (2, 2, 2), 'strides': 1, 'padding':'valid',
                             'pool': False, 'bn': True}}

    param_fcc = {'dense_1': {'neurons': 1024, 'dropout': 0.5},
                 'dense_2': {'neurons': 256, 'dropout': 0.5}}

    Model = model_w_layers((17, 17, 17), param_conv, param_fcc, data_format="channels_last")

    # num_training_data = len(m)
    num_epochs = 5
    batch_s = 60
    history = Model.fit(a, b, batch_size=batch_s, verbose=1, epochs=num_epochs,
                        # validation_split=0.2
                        )

    pred_cnn_2 = Model.predict(a)
    # pred_m = transform_cnn_output_log_mass(pred_cnn, m)
    #
    # f1 = eval.plot_loss(history)
    # f2 = eval.plot_metric(history)
    # f3 = eval.plot_true_vs_predict(m, pred_m)

    # #Use data generators
    #
    # all_ids = np.arange(256**3)
    # halo_mass = np.load("/Users/lls/Documents/mlhalos_files/halo_mass_particles.npy")
    #
    # training_ids = np.load("/Users/lls/Documents/deep_halos_files/particles.npy")
    # p_inputs = np.load("/Users/lls/Documents/deep_halos_files/3d_inputs_particles.npy")
    #
    # training_ids = np.random.choice(range(len(p_ids)), 328, replace=False)
    # testing_ids = p_ids[~np.in1d(p_ids, training_ids)]
    #
    # params = {'dim': (17, 17, 17),
    #           'batch_size': 50,
    #           'n_channels': 1,
    #           'shuffle': True}
    #
    # training_generator = DataGenerator(training_ids, np.log10(halo_mass), **params)
    # validation_generator = DataGenerator(testing_ids, np.log10(halo_mass), **params)
    #
    #
    # model.fit_generator(generator=training_generator,
    #                     validation_data=validation_generator,
    #                     use_multiprocessing=True,
    #                     workers=6)

