import time
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.layers import Dense, Flatten#, Add
from tensorflow.keras.layers import Layer
import tensorflow as tf
from dlhalos_code_tf2 import loss_functions as lf
from dlhalos_code_tf2 import callbacks as clb
from dlhalos_code_tf2 import layers
from dlhalos_code_tf2 import data_processing as dp

import copy
from tensorflow.keras.constraints import Constraint


class CNN:
    def __init__(self, conv_params, fcc_params, model_type="regression", training_dataset=None, shuffle=True,
                 validation_dataset=None, callbacks=None, metrics=None, num_epochs=5, dim=(51, 51, 51),
                 pool_size=(2, 2, 2), initialiser=None, data_format="channels_last",
                 verbose=1, save_model=False, model_name="my_model.h5", num_gpu=1,
                 lr=0.0001, loss='mse', save_summary=False, path_summary=".", validation_freq=1, train=True,
                 compile=True, validation_steps=None, steps_per_epoch=None,
                 initial_epoch=0, pretrained_model=None, weights=None, 
                 seed=None, global_average=False):

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.input_shape = dim
        self.conv_params = conv_params
        self.fcc_params = fcc_params
        self.data_format = data_format
        self.val_freq = validation_freq
        self.initialiser = initialiser
        self.pool_size = pool_size
        self.global_average = global_average
        
        self.validation_steps = validation_steps
        self.steps_per_epoch = steps_per_epoch

        self.num_gpu = num_gpu
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.metrics = metrics
        self.lr = lr
        self.callbacks = callbacks
        self.model_type = model_type
        self.loss = loss

        self.initial_epoch = initial_epoch
        self.pretrained_model = pretrained_model
        self.weights=weights
        self.seed = seed

        self.save = save_model
        self.model_name = model_name
        self.save_summary = save_summary
        self.path_summary = path_summary
        self.shuffle = shuffle
        if self.verbose == 1:
            print(f'CNN seed={self.seed}, shuffle={self.shuffle}')

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
        history = Model.fit(self.training_dataset, validation_data=self.validation_dataset,
                            initial_epoch=self.initial_epoch, verbose=self.verbose, 
                            epochs=self.num_epochs, shuffle=self.shuffle,
                            callbacks=self.callbacks, validation_freq=self.val_freq,
                            validation_steps=self.validation_steps, steps_per_epoch=self.steps_per_epoch,
                            )
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
    
    def compile_regression_model(self):
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

        self.optimiser = keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
                                                amsgrad=True)
        Model.compile(loss=self.loss, optimizer=self.optimiser, metrics=self.metrics)
        
        return Model
    
    def compile_binary_classification_model(self):
        print("Initiating binary classification model")

        Model = self.binary_classification_model_w_layers(self.input_shape, self.conv_params, self.fcc_params,
                                                          data_format=self.data_format)
        optimiser = keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
                                          amsgrad=True)
        Model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=self.metrics)
        
        return Model

    def compile_model_multiple_gpu(self):
        #TODO: need to test this new use of tf.distribute.MirroredStrategy
        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
            print("WARNING: this has not been tested yet. strategy.scope() may be in the wrong place.")
            if self.model_type == "regression":
                Model = self.compile_regression_model()
    
            elif self.model_type == "binary_classification":
                Model = self.compile_binary_classification_model()
    
            else:
                raise NameError("Choose either regression or binary classification as model type")
        
        # print(Model.summary())
        return Model

    def compile_model_single_gpu(self):
        if self.model_type == "regression":
            Model = self.compile_regression_model()

        elif self.model_type == "binary_classification":
            Model = self.compile_binary_classification_model()

        else:
            raise NameError("Choose either regression or binary classification as model type")

        # print(Model.summary())
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

        # print(Model.summary())
        return Model

    def regression_model_w_layers(self, input_shape_box, conv_params, fcc_params, data_format="channels_last"):
        input_data = keras.Input(shape=(*input_shape_box, 1))#, dtype=tf.float64)

        x, init_w = self._model(input_data, input_shape_box, conv_params, fcc_params, data_format=data_format)

        predictions = Dense(1, **fcc_params['last'], kernel_initializer=init_w, name='prediction_layer')(x)

        model = keras.Model(inputs=input_data, outputs=predictions)
        return model

    def binary_classification_model_w_layers(self, input_shape_box, conv_params, fcc_params,
                                             data_format="channels_last"):
        input_data = keras.Input(shape=(*input_shape_box, 1))#, dtype="float32")
        x = self._model(input_data, input_shape_box, conv_params, fcc_params, data_format=data_format)
        predictions = Dense(1, activation='sigmoid', **fcc_params['last'])(x)

        model = keras.Model(inputs=input_data, outputs=predictions)
        return model

    def _model(self, input_data, input_shape_box, conv_params, fcc_params, data_format="channels_last"):
        initialiser = self.get_initialiser()
        convlayers = layers.ConvLayers(conv_params, initialiser)
        denselayers = layers.FCCLayers(fcc_params, initialiser, self.seed)

        if conv_params == {}:
            x = Flatten(data_format=data_format, input_shape=(*input_shape_box, 1))(input_data)
            x = denselayers.fcc_layers(x)

        else:
            x = convlayers.conv_layers(input_data, input_shape_box)
            if self.global_average is True:
                x = keras.layers.GlobalAveragePooling3D(data_format=self.data_format)(x)
            else:
                x = Flatten(data_format=data_format)(x)
            x = denselayers.fcc_layers(x)
        return x, initialiser   
    
    def get_initialiser(self):
        if self.initialiser == "custom":

            def my_init(shape, dtype=None, partition_info=None):
                ind = int((shape[0] - 1)/2)
                print(shape)
                if shape == (3, 3, 3, 1, 4):
                    weight_matrix = np.ones(shape) * 0.001
                    weight_matrix[ind, ind, ind] = 1
                    return K.random_normal(shape, dtype=dtype, seed=self.seed) * weight_matrix
                else:
                    return K.random_normal(shape, dtype=dtype, seed=self.seed)

            initialiser = my_init

        elif self.initialiser == "lecun_normal":
            initialiser = keras.initializers.lecun_normal(seed=self.seed)

        elif self.initialiser == "Xavier_uniform":
            initialiser = keras.initializers.glorot_uniform(seed=self.seed)
            print("Initialiser is Xavier uniform")

        elif self.initialiser == "Gaussian":
            initialiser = keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=self.seed)
            print("Initialiser is Gaussian")

        else:
            initialiser = keras.initializers.he_uniform(seed=self.seed)
            print("Initialiser is he uniform")
        
        return initialiser


class LossTrainableParams(Layer):
    def __init__(self, init_gamma=None, init_alpha=None, gamma_constraint=None, alpha_constraint=None, tanh=False,
                 **kwargs):
        # self.output_dim = output_dim
        super(LossTrainableParams, self).__init__(**kwargs)

        self.init_gamma = init_gamma
        self.constraint_gamma = gamma_constraint
        self.init_alpha = init_alpha
        self.constraint_alpha = alpha_constraint

        self.tanh = tanh

    def build(self, input_shape):
        if self.init_gamma is not None:
            # Create a trainable parameter for gamma in the Cauchy log-likelihood
            init_g = tf.constant_initializer(value=self.init_gamma)
            if self.constraint_gamma is not None:
                self.gamma = self.add_weight(name='gamma', shape=(1,), initializer=init_g, trainable=True,
                                             constraint=self.constraint_gamma)

        if self.init_alpha is not None:
            # Create a trainable parameter for alpha in the weights priors terms (or, regularizers terms)
            init_a = tf.constant_initializer(value=self.init_alpha)
            self.alpha = self.add_weight(name='alpha', shape=(1,), initializer=init_a, trainable=True,
                                         constraint=self.constraint_alpha)
        super(LossTrainableParams, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        if self.tanh is True:
            return K.tanh(x)
        else:
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

    def __init__(self, conv_params, fcc_params, model_type="regression",
                 init_alpha=None, upper_bound_alpha=2., lower_bound_alpha=0., metrics=None,
                 train_gamma=True, init_gamma=0.2, upper_bound_gamma=2., lower_bound_gamma=0.,
                 regularizer_conv=None, regularizer_dense=None, alpha_mse=0.0001, shuffle=True,
                 training_dataset=None, validation_dataset=None, validation_steps=None, steps_per_epoch=None,
                 data_format="channels_last", validation_freq=1, period_model_save="epoch", dim=(51, 51, 51),
                 lr=0.0001, pool_size=(2, 2, 2), initialiser=None, pretrained_model=None, weights=None,
                 verbose=1, num_gpu=1, seed=None, save_summary=False, path_summary=".", 
                 compile=True, train=True, num_epochs=5, lr_scheduler=True, early_stopping=False,
                 load_mse_weights=False, load_weights=None, use_tanh_n_epoch=0, use_mse_n_epoch=0, 
                 optimizer=None, initial_epoch=None, global_average=False, dtype="float32"):
        ''':param period_model_save: defaults to 'epoch'. If changed to integers, integers denote No. batches per save.'''
        
        self.path_model = path_summary
        self.regularizer_conv = regularizer_conv
        self.regularizer_dense = regularizer_dense
        self.shuffle = shuffle
        if verbose == 1:
            print(f'CauchyCNN shuffle={self.shuffle}')
        self.dtype = dtype
        dp.warn_float_casting(self.get_dataset_dtype(training_dataset), self.dtype)

        self.path_model = path_summary
        self.train_gamma = train_gamma
        self.init_gamma = init_gamma
        self.LB_gamma = lower_bound_gamma
        self.UB_gamma = upper_bound_gamma
        self.constr_gamma = Between(min_value=self.LB_gamma, max_value=self.UB_gamma)

        self.init_alpha = init_alpha
        self.LB_alpha = lower_bound_alpha
        self.UB_alpha = upper_bound_alpha
        self.constr_alpha = Between(min_value=self.LB_alpha, max_value=self.UB_alpha)

        self.alpha_mse = alpha_mse
        super(CNNCauchy, self).__init__(conv_params, fcc_params, model_type=model_type,
                                        training_dataset=training_dataset, num_epochs=num_epochs, dim=dim,
                                        pool_size=pool_size, initialiser=initialiser,
                                        data_format=data_format, verbose=verbose, num_gpu=num_gpu, 
                                        lr=lr, loss='mse', save_summary=save_summary, path_summary=path_summary, train=False,
                                        compile=False, steps_per_epoch=steps_per_epoch, pretrained_model=pretrained_model, 
                                        weights=weights, seed=seed, global_average=global_average)
        if use_mse_n_epoch > 0:
            print("Get MSE")
            self.get_mse_model(load_mse_weights, conv_params, fcc_params, model_type=model_type,
                               steps_per_epoch=steps_per_epoch, training_dataset=training_dataset, 
                               dim=dim, lr=lr, verbose=verbose,  data_format=data_format,
                               num_gpu=num_gpu, pool_size=pool_size, initialiser=initialiser,
                               save_summary=False, path_summary=path_summary, pretrained_model=pretrained_model,
                               weights=weights, num_epochs=use_mse_n_epoch,
                               global_average=global_average)

        self.optimizer = optimizer
        self.metrics = metrics
        self.save_summary = save_summary

        self.num_epochs = num_epochs
        self.load_weights = load_weights
        self.use_tanh_n_epoch = use_tanh_n_epoch
        self.lr_scheduler = lr_scheduler
        self.num_gpu = num_gpu

        self.validation_dataset = validation_dataset
        self.validation_steps = validation_steps
        self.validation_freq = validation_freq

        self.period_model_save = period_model_save

        self.mse_model = self.model
        self.compile = compile
        self.train = train
        self.early_stopping = early_stopping

        if initial_epoch is not None:
            self.initial_epoch = initial_epoch

        if self.compile is True:
            print("compiling")
            self.model = self.compile_cauchy_model(self.mse_model)
            print("done compiling")

            if self.save_summary is True:
                with open(self.path_summary + 'model_summary.txt', 'w') as fh:
                    self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

            if self.load_weights is not None:
                self.model.load_weights(self.load_weights)

            if self.train is True:
                self.model, self.history, self.trained_loss_params = self.train_cauchy_model(self.model)
                np.save(self.path_model + 'trained_loss_params.npy', self.trained_loss_params)

                if self.init_alpha is not None:
                    g = [float(a) for (a, b) in self.trained_loss_params]
                    np.save(self.path_model + 'trained_loss_gamma.npy', np.insert(g, 0, self.init_gamma))
                    a = [float(b) for (a, b) in self.trained_loss_params]
                    np.save(self.path_model + 'trained_loss_alpha.npy', np.insert(a, 0, self.init_alpha))
                else:
                    g = np.insert(self.trained_loss_params, 0, self.init_gamma)
                    np.save(self.path_model + 'trained_loss_gamma.npy', g)
                    
        self.layer_outputs, self.layers_names, self.layers_weights = self.get_layers_outputs(self.model, training_dataset)
     
    def get_dataset_dtype(self, dataset):
        box_batch, label_batch = next(iter(dataset))
        return box_batch
        
    def get_layers_outputs(self, model, training_data):
        if self.verbose == 1:
            print(f'model input dtype: {model.inputs[0].dtype})')
        box, label = list(training_data.as_numpy_iterator())[0]
        outputs = [box]
        names = ['input']
        layers_weights = []
        for layer in model.layers:
            keras_function = K.function([model.input], [layer.output])
            outputs.append(keras_function([box]))
            names.append(layer.name)
            layers_weights.append(layer.get_weights())
        return outputs, names, layers_weights
    
    def compile_cauchy_model(self, mse_model, tanh=False):
        if self.num_gpu > 1:
            #TODO: this strategy.scope may need to be outside layer definitions
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                print("WARNING: this has not been tested yet. strategy.scope() may be in the wrong place.")
                new_model, optimiser, loss_c = self.assemble_cauchy_model(mse_model, tanh=tanh)
                new_model.compile(loss=loss_c, optimizer=optimiser, metrics=self.metrics)
                return new_model
        else:
            new_model, optimiser, loss_c = self.assemble_cauchy_model(mse_model, tanh=tanh)
            new_model.compile(loss=loss_c, optimizer=optimiser, metrics=self.metrics)
            return new_model

    def assemble_cauchy_model(self, mse_model, tanh=False):
        if self.optimizer is None:
            optimiser = keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
                                              amsgrad=True)
        else:
            print("Use loaded optimizer")
            optimiser = self.optimizer

        if self.train_gamma is False:
            loss_c = lf.cauchy_selection_loss_fixed_boundary(gamma=self.init_gamma, dtype=self.dtype)
            return mse_model, optimiser, loss_c

        else:
            # Define Cauchy model
            last_layer = LossTrainableParams(init_gamma=self.init_gamma, init_alpha=self.init_alpha,
                                             gamma_constraint=self.constr_gamma, alpha_constraint=self.constr_alpha,
                                             tanh=tanh)
            predictions = last_layer(mse_model.layers[-1].output)
            new_model = keras.Model(inputs=mse_model.input, outputs=predictions)

            loss_params_layer = [layer for layer in new_model.layers if 'loss_trainable_params' in layer.name][0]

            if self.init_alpha is not None: #Super sensitive to alpha so prefer not to train alpha.
                print("These are the losses from the Cauchy model before adding regularizers:")
                print(new_model.losses)

                alpha = K.pow(10., loss_params_layer.alpha)

                def add_conv_reg(index):
                    f = lambda: self.regularizer_conv(alpha)(new_model.layers[index].kernel)
                    return new_model.add_loss(f)

                def add_dense_reg(index):
                    f = lambda: self.regularizer_dense(alpha)(new_model.layers[index].kernel)
                    return new_model.add_loss(f)

                names_layers = [layer.name for layer in new_model.layers]
                conv_layers = [s for s in names_layers if 'conv3d' in s]
                indices_conv = [i for i, item in enumerate(names_layers) if item in conv_layers]
                for i in range(len(indices_conv)):
                    add_conv_reg(indices_conv[i])

                dense_layers = [s for s in names_layers if 'dense' in s]
                indices_dense = [i for i, item in enumerate(names_layers) if item in dense_layers]
                for i in range(len(indices_dense)):
                    add_dense_reg(indices_dense[i])

            print("These are the final losses from the Cauchy model:")
            print(new_model.losses)

            loss_params_layer = [layer for layer in new_model.layers if 'loss_trainable_params' in layer.name][0]
            loss_c = lf.cauchy_selection_loss_fixed_boundary_trainable_gamma(loss_params_layer, dtype=self.dtype)

            return new_model, optimiser, loss_c

    def get_callbacks(self, layer_loss=None):
        callbacks_list = []

        # checkpoint
        filepath = self.path_model + "model/weights.{epoch:02d}.h5"
        checkpoint_call = callbacks.ModelCheckpoint(filepath, save_freq=self.period_model_save, save_weights_only=True)
        callbacks_list.append(checkpoint_call)
        
        # early stopping
        if self.early_stopping:
            estop = callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=self.verbose, restore_best_weights=True)
            callbacks_list.append(estop)
            
        # learning rate scheduler
        if self.lr_scheduler:
            # lrate = callbacks.LearningRateScheduler(self.lr_scheduler_half)
            lrate = callbacks.LearningRateScheduler(self.lr_scheduler_exponential)
            callbacks_list.append(lrate)

        if self.train_gamma is True:
            # collect weights last layer
            cbk = clb.CollectWeightCallback(layer_index=-1)
            callbacks_list.append(cbk)

            # Alpha logger
            alpha_logger = clb.RegularizerCallback(layer_loss, alpha_check=[True if self.init_alpha is not None else False][0])
            callbacks_list.append(alpha_logger)

        else:
            cbk = None

        # Record training history in log file
        csv_logger = callbacks.CSVLogger(self.path_model + "training.log", separator=',', append=True)
        callbacks_list.append(csv_logger)

        return callbacks_list, cbk

    def train_cauchy_model(self, model):
        if self.train_gamma is False:
            callbacks_list, cbk = self.get_callbacks()

        else:
            # callbacks
            loss_layer = [layer for layer in model.layers if 'loss_trainable_params' in layer.name][0]
            callbacks_list, cbk = self.get_callbacks(loss_layer)

            # Train model
            if self.use_tanh_n_epoch > 0:
                print("Training for " + str(self.use_tanh_n_epoch) + " epoch with a tanh activation in the last layer")

                # Define a different model with different last layer and the load its weights onto current model
                tanh_model = self.train_with_tanh_activation(model, callbacks=callbacks_list,
                                                             num_epochs=self.use_tanh_n_epoch)
                model.set_weights(tanh_model.get_weights())
                self.initial_epoch = self.use_tanh_n_epoch
            
            if self.verbose == 1:
                if self.init_alpha is not None:
                    print("Initial value of log-alpha is %.5f" % float(K.get_value(loss_layer.alpha)))
                print("Initial value of gamma is %.5f" % float(K.get_value(loss_layer.gamma)))
        
        if self.verbose == 1:
            print("Start training with a linear activation in the last layer")
            print(f"No. Callbacks: {len(callbacks_list)}, {self.lr_scheduler}")
        history = model.fit(self.training_dataset, validation_data=self.validation_dataset,
                            initial_epoch=self.initial_epoch,
                            verbose=self.verbose, epochs=self.num_epochs, shuffle=self.shuffle,
                            callbacks=callbacks_list, validation_freq=self.validation_freq,
                            validation_steps=self.validation_steps, steps_per_epoch=self.steps_per_epoch
                            )

        return model, history, [cbk.weights if self.train_gamma is True else None][0]

    def get_mse_model(self, load_mse_weights, conv_params, fcc_params, model_type="regression",
                      training_dataset=None, steps_per_epoch=None, data_format="channels_last", dim=(51, 51, 51),
                      lr=0.0001, pool_size=(2, 2, 2), initialiser=None, pretrained_model=None, weights=None,
                      max_queue_size=10, verbose=1, num_gpu=1,
                      save_summary=False, path_summary=".", num_epochs=0, global_average=False):
        if load_mse_weights is True:
            print("Loaded initial weights given by training for one epoch on MSE loss")
            self.model.load_weights(self.path_model + 'model/mse_weights_' + str(num_epochs) + '_epoch.hdf5')

        else:
            print("Modify CONV parameters for MSE epoch")
            conv_params2 = copy.deepcopy(conv_params)
            conv_keys = [layer for layer in conv_params2.keys()]
            for key in conv_keys:
                if 'kernel_regularizer' not in conv_params2[key]:
                    conv_params2[key]['kernel_regularizer'] = self.regularizer_conv(self.alpha_mse)

            print("Modify FCC parameters for MSE epoch")
            fcc_params2 = copy.deepcopy(fcc_params)
            layer_keys = [layer for layer in fcc_params2.keys() if layer != 'last']
            for key in layer_keys:
                if 'kernel_regularizer' in fcc_params2[key]:
                    del fcc_params2[key]['kernel_regularizer']
                if 'dropout' not in fcc_params2[key]:
                    fcc_params2[key]['dropout'] = 0.5

            m = CNN(conv_params2, fcc_params2, model_type=model_type, training_dataset=training_dataset,
                    shuffle=self.shuffle, num_epochs=num_epochs, dim=dim, pool_size=pool_size, initialiser=initialiser,
                    data_format=data_format, verbose=verbose, num_gpu=num_gpu, 
                    lr=lr, loss='mse', save_summary=save_summary,
                    path_summary=path_summary, train=True, compile=True,
                    steps_per_epoch=self.steps_per_epoch,
                    pretrained_model=pretrained_model, weights=weights, seed=self.seed, global_average=global_average)
            self.model.set_weights(m.model.get_weights())
            self.model.save_weights(self.path_model + 'model/mse_weights_' + str(num_epochs) + '_epoch.hdf5')

        self.initial_epoch = num_epochs

    def train_with_tanh_activation(self, model, callbacks=None, num_epochs=0.):
        # Define a different model with different last layer and the load its weights onto current model
        _model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)

        _last_layer = LossTrainableParams(init_gamma=self.init_gamma, init_alpha=self.init_alpha,
                                         gamma_constraint=self.constr_gamma, alpha_constraint=self.constr_alpha,
                                         tanh=True)
        _predictions = _last_layer(_model.layers[-1].output)
        _tanh_model = keras.Model(inputs=_model.input, outputs=_predictions)

        _optimiser = keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,
                                            amsgrad=True)
        _loss_params_layer = [layer for layer in _tanh_model.layers if 'loss_trainable_params' in layer.name][0]
        _loss_c = lf.cauchy_selection_loss_fixed_boundary_trainable_gamma(_loss_params_layer, dtype=self.dtype)
        _tanh_model.compile(loss=_loss_c, optimizer=_optimiser)

        _h = _tanh_model.fit(self.training_dataset,
                             validation_data=self.validation_dataset,
                             verbose=self.verbose, epochs=num_epochs,
                             shuffle=self.shuffle, callbacks=callbacks, validation_freq=self.validation_freq,
                             validation_steps=self.validation_steps, steps_per_epoch=self.steps_per_epoch
                             )
        return _tanh_model

    def lr_scheduler_half(self, epoch):
        # This function halves the learning rate every ten epochs.
        init_lr = self.lr
        if epoch < 10:
            return init_lr
        else:
            drop_rate = 0.5
            epoch_drop = 10
            return init_lr * drop_rate ** np.floor(epoch / epoch_drop)

    def lr_scheduler_exponential(self, epoch):
        # This function decays the learning rate exponentially from the 10th epoch onwards.
        init_lr = self.lr
        n = 10
        if epoch < n:
            return init_lr
        else:
            return init_lr * np.math.exp(0.05 * (n - epoch))


def likelihood_metric(y_true, y_pred):
    log_lik = lf.cauchy_selection_loss_fixed_boundary(0.2)(y_true, y_pred)
    return log_lik


# def lr_scheduler(epoch):
#     # This function decays the learning rate exponentially from the 10th epoch onwards.
#     n = 10
#     if epoch < n:
#         return 0.0001
#     else:
#         return 0.0001 * np.math.exp(0.05 * (n - epoch))
#
# def lr_scheduler_half(epoch):
#     # This function halves the learning rate every ten epochs.
#     init_lr = 0.0001
#     if epoch < 10:
#         return init_lr
#     else:
#         drop_rate = 0.5
#         epoch_drop = 10
#         return init_lr * drop_rate ** np.floor(epoch / epoch_drop)


class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}
