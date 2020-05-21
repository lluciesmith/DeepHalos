# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.layers import Input, Dense, Flatten, Add, Conv3D
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
from dlhalos_code import evaluation as eval
from dlhalos_code import loss_functions as lf
from dlhalos_code import custom_regularizers as custom_reg


class CNN:
    def __init__(self, conv_params, fcc_params, model_type="regression", training_generator=None,
                 validation_generator=None, callbacks=None, metrics=None, num_epochs=5, dim=(51, 51, 51),
                 pool_size=(2, 2, 2), initialiser=None, max_queue_size=10, data_format="channels_last",
                 use_multiprocessing=False, workers=1, verbose=1, save_model=False, model_name="my_model.h5", num_gpu=1,
                 lr=0.0001, loss='mse', save_summary=False, path_summary=".", validation_freq=1, train=True,
                 skip_connector=False, compile=True, validation_steps=None, steps_per_epoch=None,
                 initial_epoch=0, pretrained_model=None, weights=None):

        self.training_generator = training_generator
        self.validation_generator = validation_generator

        self.input_shape = dim
        self.conv_params = conv_params
        self.fcc_params = fcc_params
        self.data_format = data_format
        self.val_freq = validation_freq
        self.initialiser = initialiser
        self.pool_size = pool_size

        self.validation_steps = validation_steps
        self.steps_per_epoch = steps_per_epoch

        self.num_gpu = num_gpu
        self.num_epochs = num_epochs
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.verbose = verbose
        self.metrics = metrics
        self.lr = lr
        self.callbacks = callbacks
        self.model_type = model_type
        self.skip_connector = skip_connector
        self.loss = loss

        self.initial_epoch = initial_epoch
        self.pretrained_model = pretrained_model
        self.weights=weights

        self.save = save_model
        self.model_name = model_name
        self.save_summary = save_summary
        self.path_summary = path_summary

        if train is True:
            self.model, self.history = self.compile_and_fit_model()
        else:
            if compile is True:
                self.model = self.compile_model()
            else:
                self.model = self.uncompiled_model()

    def compile_and_fit_model(self):
        Model = self.compile_model()

        t0 = time.time()
        history = Model.fit_generator(generator=self.training_generator, validation_data=self.validation_generator,
                                      use_multiprocessing=self.use_multiprocessing, workers=self.workers,
                                      max_queue_size=self.max_queue_size, initial_epoch=self.initial_epoch,
                                      verbose=self.verbose, epochs=self.num_epochs, shuffle=True,
                                      callbacks=self.callbacks, validation_freq=self.val_freq,
                                      validation_steps=self.validation_steps, steps_per_epoch=self.steps_per_epoch)
        t1 = time.time()
        print("This model took " + str((t1 - t0)/60) + " minutes to train.")

        if self.save is True:
            Model.save(self.model_name)

        return Model, history

    def compile_model(self):
        if self.num_gpu == 1:
            model = self.compile_model_single_gpu()
        elif self.num_gpu > 1:
            model = self.compile_model_multiple_gpu()
        else:
            raise ValueError

        if self.save_summary is True:
            with open(self.path_summary + 'model_summary.txt', 'w') as fh:
                model.summary(print_fn=lambda x: fh.write(x + '\n'))

        return model

    def compile_model_multiple_gpu(self):
        num_gpus = self.num_gpu

        if self.model_type == "regression":
            print("Initiating regression model on multiple GPUs")
            # with tf.device('/cpu:0'):
            Model = self.regression_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                   data_format=self.data_format)
            optimiser = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
                                              amsgrad=True)

            parallel_model = multi_gpu_model(Model, gpus=num_gpus, cpu_relocation=True, cpu_merge=True)
            parallel_model.compile(optimizer=optimiser, loss='mse', metrics=self.metrics)

        elif self.model_type == "binary_classification":
            print("Initiating binary classification model on multiple GPUs")
            # with tf.device('/cpu:0'):
            Model = self.binary_classification_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                              data_format=self.data_format)
            optimiser = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
                                              amsgrad=True)

            parallel_model = multi_gpu_model(Model, gpus=num_gpus)
            parallel_model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=self.metrics)

        else:
            raise NameError("Choose either regression or binary classification as model type")

        print(Model.summary())
        return parallel_model

    def compile_model_single_gpu(self):
        if self.model_type == "regression":
            print("Initiating regression model")

            if self.pretrained_model is not None:
                print("Loading pretrained model")
                Model = self.pretrained_model
            else:
                Model = self.regression_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                       data_format=self.data_format)

            if self.weights is not None:
                print("Loading given weights onto model")
                Model.load_weights(self.weights)

            self.optimiser = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
                                              amsgrad=True)
            Model.compile(loss=self.loss, optimizer=self.optimiser, metrics=self.metrics)

        elif self.model_type == "binary_classification":
            print("Initiating binary classification model")

            Model = self.binary_classification_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                              data_format=self.data_format)
            optimiser = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
                                              amsgrad=True)
            Model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=self.metrics)

        else:
            raise NameError("Choose either regression or binary classification as model type")

        print(Model.summary())
        return Model

    def uncompiled_model(self):
        if self.model_type == "regression":
            print("Initiating regression model")

            Model = self.regression_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                   data_format=self.data_format)

        elif self.model_type == "binary_classification":
            print("Initiating binary classification model")

            Model = self.binary_classification_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                              data_format=self.data_format)

        else:
            raise NameError("Choose either regression or binary classification as model type")

        print(Model.summary())
        return Model

    def regression_model_w_layers(self, input_shape_box, conv_params, fcc_params, data_format="channels_last"):
        input_data = Input(shape=(*input_shape_box, 1))

        if self.skip_connector is True:
            x = self._model_skip_connection(input_data, input_shape_box, conv_params, fcc_params, data_format=data_format)
        else:
            x = self._model(input_data, input_shape_box, conv_params, fcc_params, data_format=data_format)

        predictions = Dense(1, **fcc_params['last'])(x)

        model = keras.Model(inputs=input_data, outputs=predictions)
        return model

    def binary_classification_model_w_layers(self, input_shape_box, conv_params, fcc_params,
                                             data_format="channels_last"):
        input_data = Input(shape=(*input_shape_box, 1))
        x = self._model(input_data, input_shape_box, conv_params, fcc_params, data_format=data_format)
        predictions = Dense(1, activation='sigmoid', **fcc_params['last'])(x)

        model = keras.Model(inputs=input_data, outputs=predictions)
        return model

    def first_convolutional_layer(self, input_data, input_shape_box=(17, 17, 17, 1), num_kernels=3,
                                  dim_kernel=(7, 7, 7), pool_size=(2, 2, 2), strides=2, padding='valid',
                                  data_format="channels_last", kernel_regularizer=None, bias_regularizer=None,
                                  activation="linear", relu=True, alpha_relu=0.03,
                                  bn=True, pool=True, initialiser="normal"):

        x = keras.layers.Conv3D(num_kernels, dim_kernel, activation=activation,
                                strides=strides, padding=padding, data_format=data_format,
                                input_shape=input_shape_box, kernel_initializer=initialiser,
                                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(input_data)
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
        else:
            pass
        return x

    def subsequent_convolutional_layer(self, x, num_kernels=3, dim_kernel=(3, 3, 3), pool_size=(2, 2, 2), strides=2,
                                       padding='valid', kernel_regularizer=None, data_format="channels_last",
                                       activation="linear", relu=True, alpha_relu=0.03,
                                       bn=True, bias_regularizer=None,
                                       pool=True, initialiser="normal"):
        x = keras.layers.Conv3D(num_kernels, dim_kernel, activation=activation,
                                strides=strides, padding=padding, data_format=data_format,
                                kernel_initializer=initialiser, kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer)(x)
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
        else:
            pass
        return x

    def _conv_layers(self, input_data, input_shape_box, conv_params, initialiser):

        num_convolutions = len(conv_params)

        x = self.first_convolutional_layer(input_data, input_shape_box=(*input_shape_box, 1),
                                           initialiser=initialiser, **conv_params['conv_1'])
        if num_convolutions > 1:
            for i in range(1, num_convolutions):
                params = conv_params['conv_' + str(i + 1)]
                x = self.subsequent_convolutional_layer(x, initialiser=initialiser, **params)

        return x

    def subsequent_fcc_layer(self, x, neurons=1, kernel_regularizer=None, alpha_relu=0.03, activation="linear",
                             relu=True, bn=True,
                             bias_regularizer=None,  initialiser="normal", dropout=None, alpha_dropout=None):

        x = Dense(neurons, activation=activation,
                  kernel_initializer=initialiser, kernel_regularizer=kernel_regularizer,
                  bias_regularizer=bias_regularizer)(x)

        if bn is True:
            x = keras.layers.BatchNormalization(axis=-1)(x)

        if relu is True:
            print("leaky relu")
            x = keras.layers.LeakyReLU(alpha=alpha_relu)(x)

        if dropout is not None:
            x = keras.layers.Dropout(dropout)(x)

        if alpha_dropout is not None:
            x = keras.layers.AlphaDropout(alpha_dropout)(x)

        return x

    def _fcc_layers(self, x, fcc_params, initialiser):
        num_fully_connected = len(fcc_params)
        ind = range(num_fully_connected - 1)

        if num_fully_connected > 1:
            for i in ind:
                params = fcc_params['dense_' + str(i + 1)]
                x = self.subsequent_fcc_layer(x, initialiser=initialiser, **params)

        return x

    def _model(self, input_data, input_shape_box, conv_params, fcc_params, data_format="channels_last"):
        if self.initialiser == "custom":

            def my_init(shape, dtype=None, partition_info=None):
                ind = int((shape[0] - 1)/2)
                print(shape)
                if shape == (3, 3, 3, 1, 4):
                    weight_matrix = np.ones(shape) * 0.001
                    weight_matrix[ind, ind, ind] = 1
                    return K.random_normal(shape, dtype=dtype) * weight_matrix
                else:
                    return K.random_normal(shape, dtype=dtype)

            initialiser = my_init

        elif self.initialiser == "lecun_normal":
            initialiser = keras.initializers.lecun_normal()

        else:
            initialiser = keras.initializers.he_uniform()
            print("Initialiser is he uniform")

        if conv_params == {}:
            x = Flatten(data_format=data_format, input_shape=(*input_shape_box, 1))(input_data)
            x = self._fcc_layers(x, fcc_params, initialiser)

        else:
            x = self._conv_layers(input_data, input_shape_box, conv_params, initialiser)
            x = Flatten(data_format=data_format)(x)
            x = self._fcc_layers(x, fcc_params, initialiser)
        return x

    def _model_skip_connection(self, input_data, input_shape_box, conv_params, fcc_params, data_format="channels_last"):
        print("model with skip connectior")
        initialiser = keras.initializers.he_uniform()
        x_shortcut = input_data

        num_convolutions = len(conv_params)
        x = self.first_convolutional_layer(input_data, input_shape_box=(*input_shape_box, 1),
                                           initialiser=initialiser, **conv_params['conv_1'])
        if conv_params['conv_1']['pool'] is True:
            x_shortcut = keras.layers.Conv3D(conv_params['conv_1']['num_kernels'], (1, 1, 1), strides=2, padding="same",
                                             data_format=data_format, kernel_initializer=initialiser)(x_shortcut)
        if conv_params['conv_1']['bn'] is True:
            x_shortcut = keras.layers.BatchNormalization(axis=-1)(x_shortcut)

        if num_convolutions > 1:
            for i in range(1, num_convolutions):
                if i == range(1, num_convolutions)[-1]:

                    # last layer apply ReLU after merging the convolutional output and the input layers
                    params = conv_params['conv_' + str(i + 1)]
                    x = self.subsequent_convolutional_layer(x, initialiser=initialiser, **params, activation=False)

                    if params['pool'] is True:
                        x_shortcut = keras.layers.Conv3D(params['num_kernels'], (1, 1, 1), strides=2, padding="same",
                                            data_format=data_format, kernel_initializer=initialiser)(x_shortcut)
                    if params['bn'] is True:
                        x_shortcut = keras.layers.BatchNormalization(axis=-1)(x_shortcut)

                    x = Add()([x, x_shortcut])
                    x = keras.layers.LeakyReLU(alpha=0.03)(x)

                else:
                    params = conv_params['conv_' + str(i + 1)]
                    x = self.subsequent_convolutional_layer(x, initialiser=initialiser, **params)
                    if params['pool'] is True:
                        x_shortcut = keras.layers.Conv3D(params['num_kernels'], (1, 1, 1), strides=2, padding="same",
                                                data_format=data_format, kernel_initializer=initialiser)(x_shortcut)
                    if params['bn'] is True:
                        x_shortcut = keras.layers.BatchNormalization(axis=-1)(x_shortcut)

        x = Flatten(data_format=data_format)(x)
        x = self._fcc_layers(x, fcc_params, initialiser)
        return x


class LossTrainableParams(Layer):
    def __init__(self, init_gamma=0.2, init_alpha=None, gamma_constraint=None, alpha_constraint=None,
                 layers_model=None, **kwargs):
        # self.output_dim = output_dim
        super(LossTrainableParams, self).__init__(**kwargs)
        self.init_gamma = init_gamma
        self.constraint_gamma = gamma_constraint
        self.init_alpha = init_alpha
        self.constraint_alpha = alpha_constraint
        self.layers_model = layers_model

    def build(self, input_shape):
        if self.init_gamma is not None:
            # Create a trainable parameter for gamma in the Cauchy log-likelihood
            init_g = tf.constant_initializer(value=self.init_gamma)
            self.gamma = self.add_weight(name='gamma', shape=(1,), initializer=init_g, trainable=True,
                                         constraint=self.constraint_gamma)

        if self.init_alpha is not None:
            # Create a trainable parameter for alpha in the weights priors terms (or, regularizers terms)
            init_a = tf.constant_initializer(value=self.init_alpha)
            self.alpha = self.add_weight(name='alpha', shape=(1,), initializer=init_a, trainable=True,
                                         constraint=self.constraint_alpha)

        super(LossTrainableParams, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # print("Making the regularizer parameter a trainable parameter")
        # reg_losses = 0.
        # for layer in new_model.layers[:-2]:
        #     if 'conv3d' in layer.name:
        #         print(layer)
        #         reg_losses += custom_reg.l2_norm(loss_params_layer.alpha)(layer.kernel)
        #     elif 'dense' in layer.name:
        #         print(layer)
        #         reg_losses += custom_reg.l1_and_l21_group(loss_params_layer.alpha)(layer.kernel)
        #     else:
        #         pass

        # for layer in self.layers_model[:-1]:
        #     if 'conv3d' in layer.name:
        #         print(layer)
        #         layer.add_loss(lambda: self.alpha * custom_reg.l2_norm(1.)(layer.kernel))
        #     elif 'dense' in layer.name:
        #         print(layer)
        #         layer.add_loss(lambda: self.alpha * custom_reg.l1_and_l21_group(1.)(layer.kernel))
        #     else:
        #         pass
        return x

    def get_config(self):
        return {'alpha': self.alpha, 'gamma':self.gamma}


class CNNCauchy(CNN):
    """
    This is the model which uses the Cauchy+selection+fixed boundary loss,
    after training the CNN on MSE loss for one epoch.

    Important note: If you want to train the regularization parameter, alpha,
    you must provide as input a `LossTrainableParams` layer which contains the parameter alpha
    that was used in defining the regularizers inside `conv_params' and `fcc_params'.

    """

    def __init__(self, conv_params, fcc_params, model_type="regression", init_gamma=0.2, init_alpha=None,
                 upper_bound_alpha=2., lower_bound_alpha=0., upper_bound_gamma=2., lower_bound_gamma=0.,
                 training_generator=None, validation_generator=None, validation_steps=None, steps_per_epoch=None,
                 data_format="channels_last", validation_freq=1, period_model_save=1, dim=(51, 51, 51),
                 lr=0.0001, pool_size=(2, 2, 2), initialiser=None, pretrained_model=None, weights=None,
                 max_queue_size=10, use_multiprocessing=False, workers=1, verbose=1, num_gpu=1,
                 save_summary=False, path_summary=".", compile=True, train=True,
                 load_mse_weights=False, num_epochs=5, initial_epoch=1, load_weights=None):

        self.path_model = path_summary


        # initialize CNN to load/train weights for one epoch on MSE
        train_bool = not load_mse_weights
        super(CNNCauchy, self).__init__(conv_params, fcc_params, model_type=model_type,
                                        steps_per_epoch=steps_per_epoch,
                                        training_generator=training_generator, dim=dim,
                                        loss='mse', num_epochs=1, lr=lr, verbose=verbose, data_format=data_format,
                                        use_multiprocessing=use_multiprocessing, workers=workers, num_gpu=num_gpu,
                                        pool_size=pool_size, initialiser=initialiser, save_summary=save_summary,
                                        path_summary=path_summary, pretrained_model=pretrained_model,
                                        weights=weights,
                                        max_queue_size=max_queue_size, train=train_bool)
        if train_bool is False:
            print("Loaded initial weights given by training for one epoch on MSE loss")
            self.model.load_weights(self.path_model + 'model/mse_weights_one_epoch.hdf5')
        else:
            self.model.save_weights(self.path_model + 'model/mse_weights_one_epoch.hdf5')

        self.init_gamma = init_gamma
        self.LB_gamma = lower_bound_gamma
        self.UB_gamma = upper_bound_gamma

        self.init_alpha = init_alpha
        self.LB_alpha = lower_bound_alpha
        self.UB_alpha = upper_bound_alpha

        self.num_epochs = num_epochs
        self.initial_epoch = initial_epoch
        self.load_weights = load_weights

        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.validation_freq = validation_freq

        self.period_model_save = period_model_save

        self.compile = compile
        self.train = train
        print(self.compile)
        print(self.train)

        if self.compile is True:
            self.model = self.compile_cauchy_model(self.model)

            if self.load_weights is not None:
                self.model.load_weights(self.load_weights)

            if self.train is True:
                self.model, self.history, self.trained_loss_params = self.train_cauchy_model(self.model)
                np.save(self.path_model + 'trained_loss_params.npy', self.trained_loss_params)

                # if self.init_alpha is None:
                #     g = np.insert(self.trained_loss_params, 0, self.init_gamma)
                #     np.save(self.path_model + 'trained_loss_gamma.npy', g)
                # else:
                #     g, a = self.trained_loss_params
                #     np.save(self.path_model + 'trained_loss_gamma.npy', np.insert(g, 0, self.init_gamma))
                #     np.save(self.path_model + 'trained_loss_alpha.npy', np.insert(a, 0, self.init_alpha))

    def compile_cauchy_model(self, mse_model):
        # Define Cauchy model
        reg_gamma = tf.keras.constraints.MinMaxNorm(min_value=self.LB_gamma, max_value=self.UB_gamma, rate=1.0, axis=0)
        reg_alpha = tf.keras.constraints.MinMaxNorm(min_value=self.LB_alpha, max_value=self.UB_alpha, rate=1.0, axis=0)
        last_layer = LossTrainableParams(init_gamma=self.init_gamma, init_alpha=self.init_alpha,
                                         gamma_constraint=reg_gamma, alpha_constraint=reg_alpha,
                                         layers_model=mse_model.layers)

        predictions = last_layer(mse_model.layers[-1].output)
        new_model = keras.Model(inputs=mse_model.input, outputs=predictions)

        # We have to modify the form of the regularizers to take alpha as a trainable parameter
        names = [layer.name for layer in new_model.layers]
        for i, name in enumerate(names):
            if 'loss_trainable_params' in name:
                loss_params_layer = new_model.layers[i]
        #
        # reg_losses = None
        # if self.init_alpha is not None:
        #     print("Making the regularizer parameter a trainable parameter")
        #     # reg_losses = 0.
        #     # for layer in new_model.layers[:-2]:
        #     #     if 'conv3d' in layer.name:
        #     #         print(layer)
        #     #         reg_losses += custom_reg.l2_norm(loss_params_layer.alpha)(layer.kernel)
        #     #     elif 'dense' in layer.name:
        #     #         print(layer)
        #     #         reg_losses += custom_reg.l1_and_l21_group(loss_params_layer.alpha)(layer.kernel)
        #     #     else:
        #     #         pass
        #
        #     for layer in new_model.layers[:-2]:
        #         if 'conv3d' in layer.name:
        #             print(layer)
        #             layer.kernel_regularizer = custom_reg.l2_norm(1)
        #         elif 'dense' in layer.name:
        #             print(layer)
        #             layer.kernel_regularizer = custom_reg.l1_and_l21_group(1)
        #         else:
        #             pass

            # new_model._losses *= last_layer.alpha
        if self.init_alpha is not None:
            for layer in new_model.layers[:-2]:
                if isinstance(layer, Conv3D):
                    print(layer)
                    # l = new_model.layers[-1].alpha * custom_reg.l2_norm(1.)(layer.kernel)
                    new_model.add_loss(loss_params_layer.alpha * custom_reg.l2_norm(1.)(layer.kernel))
                if isinstance(layer, Dense):
                    print(layer)
                    new_model.add_loss(loss_params_layer.alpha * custom_reg.l1_and_l21_group(1.)(layer.kernel))
                else:
                    pass
        print(new_model.losses)

        optimiser = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        loss_c = lf.cauchy_selection_loss_fixed_boundary_trainable_gamma(last_layer)

        new_model.compile(loss=loss_c, optimizer=optimiser)
        return new_model

    def train_cauchy_model(self, model):

        # callbacks
        filepath = self.path_model + "/model/weights.{epoch:02d}.hdf5"
        checkpoint_call = callbacks.ModelCheckpoint(filepath, period=self.period_model_save, save_weights_only=True)
        lrate = callbacks.LearningRateScheduler(lr_scheduler_half)
        cbk = CollectWeightCallback(layer_index=-1)
        csv_logger = callbacks.CSVLogger(self.path_model + "/training.log", separator=',', append=True)
        # callbacks_list = [checkpoint_call, csv_logger, lrate, cbk]
        alpha_logger = RegularizerCallback(model.layers[-1])
        callbacks_list = [checkpoint_call, csv_logger, lrate, cbk, alpha_logger]

        # Train model

        history = model.fit_generator(generator=self.training_generator, validation_data=self.validation_generator,
                                      use_multiprocessing=self.use_multiprocessing, workers=self.workers,
                                      max_queue_size=self.max_queue_size, initial_epoch=self.initial_epoch,
                                      verbose=self.verbose, epochs=self.num_epochs, shuffle=True,
                                      callbacks=callbacks_list, validation_freq=self.val_freq,
                                      validation_steps=self.validation_steps, steps_per_epoch=self.steps_per_epoch)

        return model, history, cbk.weights


class RegularizerCallback(Callback):
    def __init__(self, layer):
        super(Callback, self).__init__()
        self.layer = layer

    def on_train_batch_end(self, batch, logs=None):
        print("Updated alpha to value %.5f" % float(K.get_value(self.layer.alpha)))
        print("Updated gamma to value %.5f" % float(K.get_value(self.layer.gamma)))


def lr_scheduler_half(epoch):
    # This function halves the learning rate every ten epochs.
    init_lr = 0.0001
    if epoch < 10:
        return init_lr
    else:
        drop_rate = 0.5
        epoch_drop = 10
        return init_lr * drop_rate**np.floor(epoch / epoch_drop)


def lr_scheduler(epoch):
    # This function decays the learning rate exponentially from the 10th epoch onwards.
    n = 10
    if epoch < n:
        return 0.0001
    else:
        return 0.0001 * np.math.exp(0.05 * (n - epoch))


class CollectWeightCallback(Callback):
    def __init__(self, layer_index):
        super(CollectWeightCallback, self).__init__()
        self.layer_index = layer_index
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.layers[self.layer_index]
        self.weights.append(layer.get_weights())


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



