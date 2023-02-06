import tensorflow as tf
import tensorflow.keras as keras
from dlhalos_code_tf2 import loss_functions as losses


class ConvLayers:
    def __init__(self, conv_params, initialiser):
        self.conv_params = conv_params
        self.initialiser = initialiser

    def conv_layers(self, input_layer, input_shape_box):
        num_convolutions = len(self.conv_params)

        x = self.first_convolutional_layer(input_layer, input_shape_box=(*input_shape_box, 1),
                                           initialiser=self.initialiser, **self.conv_params['conv_1'])
        if num_convolutions > 1:
            for i in range(1, num_convolutions):
                params = self.conv_params['conv_' + str(i + 1)]
                x = self.subsequent_convolutional_layer(x, initialiser=self.initialiser, **params)

        return x

    def first_convolutional_layer(self, input_data, input_shape_box=(17, 17, 17, 1), num_kernels=3,
                                  dim_kernel=(7, 7, 7), pool_size=(2, 2, 2), strides=2, padding='valid',
                                  data_format="channels_last", kernel_regularizer=None, bias_regularizer=None,
                                  activation="linear", relu=True, alpha_relu=0.03,
                                  bn=True, pool="max", initialiser="normal"):

        x = keras.layers.Conv3D(num_kernels, dim_kernel, activation=activation,
                                strides=strides, padding=padding, data_format=data_format,
                                input_shape=input_shape_box, kernel_initializer=initialiser,
                                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(input_data)
        x = self._post_convolution(x, bn=bn, relu=relu, alpha_relu=alpha_relu, pool=pool, pool_size=pool_size,
                                   data_format=data_format)
        return x

    def subsequent_convolutional_layer(self, x, num_kernels=3, dim_kernel=(3, 3, 3), pool_size=(2, 2, 2), strides=2,
                                       padding='valid', kernel_regularizer=None, data_format="channels_last",
                                       activation="linear", relu=True, alpha_relu=0.03,
                                       bn=True, bias_regularizer=None,
                                       pool="max", initialiser="normal"):
        x = keras.layers.Conv3D(num_kernels, dim_kernel, activation=activation,
                                strides=strides, padding=padding, data_format=data_format,
                                kernel_initializer=initialiser, kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer)(x)
        x = self._post_convolution(x, bn=bn, relu=relu, alpha_relu=alpha_relu, pool=pool, pool_size=pool_size,
                                   data_format=data_format)
        return x

    def _post_convolution(self, x, bn=False, relu=True, alpha_relu=0.03, pool="max", pool_size=(2, 2, 2),
                          data_format="channels_last"):
        if bn is True:
            x = keras.layers.BatchNormalization(axis=-1)(x)

        if relu is True:
            x = keras.layers.LeakyReLU(alpha=alpha_relu)(x)

        if pool == "average":
            x = keras.layers.AveragePooling3D(pool_size=pool_size, strides=None, padding="same",
                                              data_format=data_format)(x)
        elif pool == "max":
            x = keras.layers.MaxPooling3D(pool_size=pool_size, strides=None, padding="same",
                                          data_format=data_format)(x)
            print('adding max pooling')
        else:
            pass
        return x


class FCCLayers:
    def __init__(self, fcc_params, initialiser, seed=None):
        self.fcc_params = fcc_params
        self.initialiser = initialiser
        self.seed = seed

    def fcc_layers(self, x, name_1st_layer=None):
        num_fully_connected = len(self.fcc_params)
        ind = range(num_fully_connected - 1)

        if num_fully_connected > 1:
            for i in ind:
                params = self.fcc_params['dense_' + str(i + 1)]
                if i == 0:
                    x = self.subsequent_fcc_layer(x, name=name_1st_layer, initialiser=self.initialiser, **params)
                else:
                    x = self.subsequent_fcc_layer(x, initialiser=self.initialiser, **params)

        return x

    def subsequent_fcc_layer(self, x, neurons=1, kernel_regularizer=None, alpha_relu=0.03, activation="linear",
                             relu=True, bn=True, bias_regularizer=None, initialiser="normal", dropout=None,
                             alpha_dropout=None, name=None):
        x = keras.layers.Dense(neurons, activation=activation, kernel_initializer=initialiser, name=name,
                               kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(x)

        if bn is True:
            x = keras.layers.BatchNormalization(axis=-1)(x)

        if relu is True:
            print("leaky relu")
            x = keras.layers.LeakyReLU(alpha=alpha_relu)(x)

        if dropout is not None:
            print("using dropout")
            x = keras.layers.Dropout(dropout, seed=self.seed)(x)

        if alpha_dropout is not None:
            # keeps mean and variance of inputs to their original values,
            # in order to ensure the self-normalizing property even after this dropout.
            x = keras.layers.AlphaDropout(alpha_dropout)(x)

        return x


class KLLossLayer(keras.layers.Layer):
    def __init__(self, beta0, name="KLlayer"):
        super(KLLossLayer, self).__init__(name=name)
        self.beta0 = beta0
        self.beta = tf.Variable(initial_value=beta0, trainable=False)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_term = losses.KL_loss(z_mean, z_log_var)
        self.add_loss(self.beta * kl_term)
        self.add_metric(kl_term, name="KL")
        return inputs

    def get_config(self):
        return {"beta0": self.beta0}