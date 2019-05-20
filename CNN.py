import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow.keras.models as Model
from tensorflow.keras.layers import Input, Dense, Conv3D, Flatten
import evaluation as eval


def preprocess_data(input_data, output_label, normalise_output=True):
    """ input is reshaped and output is normalised """

    input_processed = input_data.reshape(input_data.shape[0], input_data.shape[1], input_data.shape[2],
                                         input_data.shape[3], 1)
    if normalise_output is True:
        output_new = (output_label - min(output_label)) / (max(output_label) - min(output_label))
    else:
        output_new = output_label

    output_processed = output_new.reshape(-1, 1)
    return input_processed, output_processed


def transform_cnn_output_log_mass(prediction, m):
    return (prediction * (max(m) - min(m))) + min(m)


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


def model_w_layers(input_shape_box=(17, 17, 17, 1), batch_size=1, data_format="channels_last", num_conv=1,
                   num_kernels=3, dim_kernel=(7, 7, 7), strides=2,  padding='valid',
                   alpha_relu=0.3, activation=True, bn=True, pool=True):

    input_data = Input(shape=input_shape_box, batch_size=batch_size)

    # create `num_conv` 3D convolutional layer(s) followed by activation, batch normalisation and average pooling

    if num_conv == 1:
        num_kernels = [num_kernels]
        strides = [strides]
        dim_kernel = [dim_kernel]

    assert (len(num_kernels), len(dim_kernel), len(strides)) == (num_conv, num_conv, num_conv)

    # create a 3D convolutional layer followed by activation , batch normalisation and pooling

    x = first_convolutional_layer(input_data, input_shape_box=input_shape_box,
                                  num_kernels=num_kernels[0], dim_kernel=dim_kernel[0], strides=strides[0],
                                  padding=padding, data_format=data_format, alpha_relu=alpha_relu,
                                  activation=activation, bn=bn, pool=pool)
    if num_conv == 1:
        pass
    else:
        for i in range(1, num_conv):
            x = subsequent_convolutional_layer(x, num_kernels=num_kernels[i], dim_kernel=dim_kernel[i],
                                               strides=strides[i], padding=padding, data_format=data_format,
                                               alpha_relu=alpha_relu, activation=activation, bn=bn, pool=pool)

    # Flatten and fully connected layers

    x = Flatten(data_format=data_format)(x)
    x = Dense(8, activation='relu')(x)
    predictions = Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input_data, outputs=predictions)

    model.compile(optimizer='adam', loss='mse', metrics=["mae"])
    return model


if __name__ == "__main__":

    # Load inputs

    p_ids = np.load("/Users/lls/Documents/deep_halos_files/particles.npy")

    p_inputs = np.load("/Users/lls/Documents/deep_halos_files/3d_inputs_particles.npy")
    m = np.load("/Users/lls/Documents/deep_halos_files/outputs_particles.npy")
    a, b = preprocess_data(p_inputs, m, normalise_output=True)

    num_training_data = len(m)
    num_epochs = 5
    batch_s = int(len(m)/5)

    seed = 7
    np.random.seed(seed)

    num_convolutions = 2
    num_kernels = [10, 15]
    dim_kernel = [(7, 7, 7), (3, 3, 3)]
    strides = [2, 2]

    Model = model_w_layers(input_shape_box=(17, 17, 17, 1), batch_size=batch_s,
                           num_conv=num_convolutions, num_kernels=num_kernels, dim_kernel=dim_kernel, strides=strides,
                           padding='valid', data_format="channels_last", alpha_relu=0.3, pool=False)

    history = Model.fit(a, b, batch_size=batch_s, verbose=1, epochs=num_epochs,
                        validation_split=0.2)

    pred_cnn = Model.predict(a)
    pred_m = transform_cnn_output_log_mass(pred_cnn, m)

    f1 = eval.plot_loss(history)
    f2 = eval.plot_metric(history)
    f3 = eval.plot_true_vs_predict(m, pred_m)

