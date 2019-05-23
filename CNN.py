# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv3D, Flatten
from tensorflow import set_random_seed
import sklearn.preprocessing
import time
from tensorflow.keras.utils import plot_model


class CNN:
    def __init__(self, training_generator, validation_generator, conv_params, fcc_params, num_epochs=5,
                 data_format="channels_last", use_multiprocessing=False, workers=1, verbose=1, save=False):

        self.training_generator = training_generator
        self.validation_generator = validation_generator

        self.input_shape = training_generator.dim
        self.conv_params = conv_params
        self.fcc_params = fcc_params
        self.data_format = data_format

        self.num_epochs = num_epochs
        self.use_multiprocessing = use_multiprocessing
        self.workers = workers
        self.verbose = verbose

        Model = self.model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                         data_format=self.data_format)
        print(Model.summary())
        t0 = time.time()
        history = Model.fit_generator(generator=self.training_generator, validation_data=self.validation_generator,
                                      use_multiprocessing=self.use_multiprocessing, workers=self.workers,
                                      verbose=self.verbose, epochs=self.num_epochs, shuffle=True)
        t1 = time.time()
        print("This model took " + str((t1 - t0)/60) + " minutes to train.")

        if save is True:
            plot_model(Model, to_file='model.png')

        self.model = Model
        self.history = history

    def first_convolutional_layer(self, input_data, input_shape_box=(17, 17, 17, 1),
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

    def subsequent_convolutional_layer(self, x, num_kernels=3, dim_kernel=(7, 7, 7), strides=2, padding='valid',
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

    def model_w_layers(self, input_shape_box, conv_params, fcc_params, data_format="channels_last"):

        input_data = Input(shape=(*input_shape_box, 1))
        num_convolutions = len(conv_params)
        num_fully_connected = len(fcc_params)

        x = self.first_convolutional_layer(input_data, input_shape_box=input_shape_box, **conv_params['conv_1'])

        if num_convolutions > 1:
            for i in range(1, num_convolutions):
                params = conv_params['conv_' + str(i + 1)]
                x = self.subsequent_convolutional_layer(x, **params)

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


