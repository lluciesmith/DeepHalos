# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow.keras as keras
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv3D, Flatten
import time
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import normal
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback
import numpy as np
import evaluation as eval


class CNN:
    def __init__(self, training_generator, conv_params, fcc_params, model_type="regression",
                 validation_generator=None, callbacks=None, metrics=None, num_epochs=5,
                 data_format="channels_last", use_multiprocessing=False, workers=1, verbose=1, save=False,
                 model_name="my_model.h5", num_gpu=1):

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
        self.metrics = metrics
        self.callbacks = callbacks
        self.model_type = model_type

        self.save = save
        self.model_name = model_name

        if num_gpu == 1:
            self.model, self.history = self.fit_model_single_gpu()
        elif num_gpu > 1:
            self.model, self.history = self.fit_model_multiple_gpu(num_gpu)

    def fit_model_single_gpu(self):
        if self.model_type == "regression":
            print("Initiating regression model")

            Model = self.regression_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                   data_format=self.data_format)

            optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                              amsgrad=True)
            Model.compile(optimizer=optimiser, loss='mse', metrics=self.metrics)

        elif self.model_type == "binary_classification":
            print("Initiating binary classification model")

            Model = self.binary_classification_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                              data_format=self.data_format)
            optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                              amsgrad=True)
            Model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=self.metrics)

        else:
            raise NameError("Choose either regression or binary classification as model type")

        print(Model.summary())
        t0 = time.time()
        history = Model.fit_generator(generator=self.training_generator, validation_data=self.validation_generator,
                                      use_multiprocessing=self.use_multiprocessing, workers=self.workers,
                                      verbose=self.verbose, epochs=self.num_epochs, shuffle=True,
                                      callbacks=self.callbacks)
        t1 = time.time()
        print("This model took " + str((t1 - t0)/60) + " minutes to train.")

        if self.save is True:
            Model.save(self.model_name)

        return Model, history

    def fit_model_multiple_gpu(self, num_gpus):

        if self.model_type == "regression":
            with tf.device('/cpu:0'):
                Model = self.regression_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                       data_format=self.data_format, metrics=self.metrics)
                parallel_model = multi_gpu_model(Model, gpus=num_gpus)
                optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                                  amsgrad=True)
                parallel_model.compile(optimizer=optimiser, loss='mse', metrics=self.metrics)

        elif self.model_type == "binary_classification":
            with tf.device('/cpu:0'):
                print("Initiating binary classification model")
                Model = self.binary_classification_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                              data_format=self.data_format, metrics=self.metrics)
                parallel_model = multi_gpu_model(Model, gpus=num_gpus)
                optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                                  amsgrad=True)
                parallel_model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=self.metrics)

        else:
            NameError("Choose either regression or binary classification as model type")

        t0 = time.time()
        history = parallel_model.fit_generator(generator=self.training_generator,
                                               validation_data=self.validation_generator,
                                               use_multiprocessing=self.use_multiprocessing, workers=self.workers,
                                               verbose=self.verbose, epochs=self.num_epochs, shuffle=True,
                                               callbacks=self.callbacks)
        t1 = time.time()
        print("This model took " + str((t1 - t0)/60) + " minutes to train.")

        if self.save is True:
            Model.save(self.model_name)

        return parallel_model, history

    def first_convolutional_layer(self, input_data, input_shape_box=(17, 17, 17, 1), num_kernels=3,
                                  dim_kernel=(7, 7, 7), strides=2, padding='valid', data_format="channels_last",
                                  alpha_relu=0.3, activation=True, bn=True, pool=True, initialiser="normal"):

        x = keras.layers.Conv3D(num_kernels, dim_kernel, strides=strides, padding=padding, data_format=data_format,
                                input_shape=input_shape_box, kernel_initializer=initialiser)(input_data)
        if activation is True:
            x = keras.layers.LeakyReLU(alpha=alpha_relu)(x)
        if bn is True:
            x = keras.layers.BatchNormalization(axis=-1)(x)
        if pool is True:
            x = keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding=padding,
                                              data_format=data_format)(x)
        return x

    def subsequent_convolutional_layer(self, x, num_kernels=3, dim_kernel=(7, 7, 7), strides=2, padding='valid',
                                       data_format="channels_last", alpha_relu=0.3, activation=True, bn=True, pool=True,
                                       initialiser="normal"):
        x = keras.layers.Conv3D(num_kernels, dim_kernel, strides=strides, padding=padding, data_format=data_format,
                                kernel_initializer=initialiser)(x)
        if activation is True:
            x = keras.layers.LeakyReLU(alpha=alpha_relu)(x)
        if bn is True:
            x = keras.layers.BatchNormalization(axis=-1)(x)
        if pool is True:
            x = keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding=padding,
                                              data_format=data_format)(x)
        return x

    def regression_model_w_layers(self, input_shape_box, conv_params, fcc_params, data_format="channels_last"):

        initialiser = tf.compat.v1.keras.initializers.TruncatedNormal()

        input_data = Input(shape=(*input_shape_box, 1))
        num_fully_connected = len(fcc_params)

        if conv_params == {}:
            x = Flatten(data_format=data_format, input_shape=(*input_shape_box, 1))(input_data)

            if num_fully_connected > 1:
                for i in range(num_fully_connected):
                    params = fcc_params['dense_' + str(i + 1)]
                    x = Dense(params['neurons'], activation='relu', kernel_initializer=initialiser)(x)
                    if "dropout" in params:
                        x = keras.layers.Dropout(params['dropout'])(x)

            predictions = Dense(1, activation='linear')(x)

        else:
            num_convolutions = len(conv_params)
            num_fully_connected = len(fcc_params)

            x = self.first_convolutional_layer(input_data, input_shape_box=(*input_shape_box, 1),
                                               initialiser=initialiser,
                                               **conv_params['conv_1'])

            if num_convolutions > 1:
                for i in range(1, num_convolutions):
                    params = conv_params['conv_' + str(i + 1)]
                    x = self.subsequent_convolutional_layer(x, initialiser=initialiser, **params)

            # Flatten and fully connected layers, followed by dropout

            x = Flatten(data_format=data_format)(x)

            if num_fully_connected > 1:
                for i in range(num_fully_connected):
                    params = fcc_params['dense_' + str(i + 1)]
                    x = Dense(params['neurons'], activation='relu', kernel_initializer=initialiser)(x)
                    if "dropout" in params:
                        x = keras.layers.Dropout(params['dropout'])(x)

            predictions = Dense(1, activation='linear')(x)

        model = keras.Model(inputs=input_data, outputs=predictions)
        return model

    def binary_classification_model_w_layers(self, input_shape_box, conv_params, fcc_params,
                                             data_format="channels_last"):

        # initialiser = tf.compat.v1.keras.initializers.TruncatedNormal()
        initialiser = keras.initializers.he_uniform()
        # initialiser = normal(mean=0, stddev=0.1, seed=13)

        input_data = Input(shape=(*input_shape_box, 1))
        num_convolutions = len(conv_params)
        num_fully_connected = len(fcc_params)

        x = self.first_convolutional_layer(input_data, input_shape_box=(*input_shape_box, 1), initialiser=initialiser,
                                           ** conv_params['conv_1'])

        if num_convolutions > 1:
            for i in range(1, num_convolutions):
                params = conv_params['conv_' + str(i + 1)]
                x = self.subsequent_convolutional_layer(x, initialiser=initialiser, **params)

        # Flatten and fully connected layers, followed by dropout

        x = Flatten(data_format=data_format)(x)

        if num_fully_connected > 1:
            for i in range(num_fully_connected):
                params = fcc_params['dense_' + str(i + 1)]
                x = Dense(params['neurons'], activation='relu', kernel_initializer=initialiser)(x)
                if "dropout" in params:
                    x = keras.layers.Dropout(params['dropout'])(x)

        predictions = Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=input_data, outputs=predictions)
        return model


class AucCallback(Callback):
    def __init__(self, training_data, validation_data, name_training="0", names_val="1"):
        self.training_generator = training_data[0]
        self.labels_training = training_data[1]

        self._validation_data = validation_data

        if isinstance(validation_data, list):
            self.validation_generator = [i[0] for i in validation_data]
            self.labels_validation = [i[1] for i in validation_data]

        elif isinstance(self._validation_data, tuple):
            self.validation_generator = validation_data[0]
            self.labels_validation = validation_data[1]

        self.names_training = name_training
        self.names_val = names_val

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        name_train = "auc_train_" + str(self.names_training)
        logs[name_train] = self.get_auc(self.training_generator, self.labels_training)

        if isinstance(self._validation_data, list):
            for i in range(len(self._validation_data)):
                name_i = "auc_val_" + str(self.names_val[i])
                logs[name_i] = self.get_auc(self.validation_generator[i], self.labels_validation[i])

        elif isinstance(self._validation_data, tuple):
            name_val = "auc_val_" + str(self.names_val)
            logs[name_val] = self.get_auc(self.validation_generator, self.labels_validation)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def get_auc(self, generator, labels):
        t0 = time.time()

        y_pred = self.model.predict_generator(generator)
        y_pred_proba = np.column_stack((1 - y_pred[:, 0], y_pred[:, 0]))
        auc_score = eval.roc(y_pred_proba, labels, true_class=1, auc_only=True)

        t1 = time.time()
        print("AUC computation for a single dataset took " + str((t1 - t0) / 60) + " minutes.")
        print("AUC = %s" % auc_score)
        return auc_score


class LossCallback(Callback):
    def __init__(self, validation_generators, names_val="1"):
        self.validation_generator = validation_generators
        self.names_val = names_val

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if isinstance(self._validation_data, list):
            for i in range(len(self.validation_generator)):
                name_i = "loss_val_" + str(self.names_val[i])
                loss_i = self.model.evaluate_generator(self.validation_generator[i])
                logs[name_i] = loss_i
                print("loss = %s" % loss_i)

        elif isinstance(self._validation_data, tuple):
            name_val = "loss_val_" + str(self.names_val)
            logs[name_val] = self.model.evaluate_generator(self.validation_generator)

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return




