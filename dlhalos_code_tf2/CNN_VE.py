import time

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input, Dense, Flatten
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from dlhalos_code_tf2 import CNN
from dlhalos_code_tf2 import layers


class VCE(CNN.CNN):
    def __init__(self, latent_dim, beta, conv_params, fcc_params, model_type="regression", training_dataset=None,
                 validation_dataset=None, callbacks=None, metrics=None, num_epochs=5, dim=(51, 51, 51),
                 pool_size=(2, 2, 2), initialiser=None, data_format="channels_last",
                 verbose=1, save_model=False, model_name="my_model.h5", num_gpu=1,
                 lr=0.0001, validation_freq=1, train=True):
        super().__init__(training_dataset, conv_params, model_type=model_type,
                         validation_dataset=validation_dataset, callbacks=callbacks, metrics=save_model,
                         num_epochs=num_epochs, data_format=data_format, 
                         verbose=verbose, save_model=save_model, model_name=model_name, num_gpu=num_gpu,
                         lr=lr, validation_freq=validation_freq, train=False)

        self.beta = beta
        # self.plot_models = plot_models
        self.latent_dim = latent_dim
        input_shape = self.input_shape

        # encoder

        input_encoder = Input(shape=(*input_shape, 1), name='encoder_input')
        z_mean, z_log_var, z = self.encoder_net(input_encoder, input_shape, conv_params, latent_dim)
        encoder = Model(input_encoder, [z_mean, z_log_var, z], name='encoder')

        # decoder

        input_decoder = Input(shape=(self.latent_dim,), name='decoder_input')
        output_decoder = self.decoder_net(input_decoder, fcc_params)
        decoder = Model(input_decoder, output_decoder, name='decoder')

        # VCE

        output_vce = decoder(encoder(input_encoder)[2])
        vce = Model(input_encoder, output_vce, name='vce')

        # Compile VCE

        def vce_loss(inputs, outputs):
            reconstruction_loss = mse(inputs, outputs)
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return reconstruction_loss + self.beta * kl_loss

        vce = self.compile_vce_model(vce, loss=vce_loss)

        # Train VCE

        if train is True:
            t0 = time.time()
            history = vce.fit(generator=self.training_dataset, validation_data=self.validation_dataset,
                              verbose=self.verbose, epochs=self.num_epochs, shuffle=True,
                              callbacks=self.callbacks, validation_freq=self.val_freq)
            t1 = time.time()
            print("This model took " + str((t1 - t0)/60) + " minutes to train.")
            self.history = history

        self.vce = vce
        self.encoder = encoder
        self.decoder = decoder

    def vae_loss(self, inputs, outputs, z_mu, log_z_variance):
        reconstruction_loss = mse(inputs, outputs)
        kl_loss = -0.5 * K.mean(1 + log_z_variance - K.square(z_mu) - K.exp(log_z_variance), axis=-1)
        return reconstruction_loss + self.beta * kl_loss

    def compile_vce_model(self, vce_model, loss):
        optimiser = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        vce_model.compile(optimizer=optimiser, loss=loss, metrics=self.metrics)
        print(vce_model.summary())

        if self.plot_models is True:
            plot_model(vce_model, to_file='my_vae.png', show_shapes=True)

        return vce_model

    # def encoder_model(self, input_shape_box, conv_params, latent_dim, input_encoder=None):
    #     if input_encoder is None:
    #         input_encoder = Input(shape=(self.input_shape,), name='encoder_input')
    #
    #     z_mean, z_log_var, z = self.encoder_net(input_encoder, input_shape_box, conv_params, latent_dim)
    #     encoder = Model(input_encoder, [z_mean, z_log_var, z], name='encoder')
    #
    #     print(encoder.summary())
    #
    #     if self.plot_models is True:
    #         plot_model(encoder, to_file='my_encoder.png', show_shapes=True)
    #     return encoder
    #
    # def decoder_model(self, fcc_params):
    #     input_decoder = Input(shape=(self.latent_dim,), name='decoder_input')
    #     output_decoder = self.decoder_net(input_decoder, fcc_params)
    #
    #     decoder = Model(input_decoder, output_decoder, name='decoder')
    #     print(decoder.summary())
    #     if self.plot_models is True:
    #         plot_model(decoder, to_file='my_decoder.png', show_shapes=True)
    #     return decoder

    def sampling(self, args):
        """
        Instead of sampling from Q(z|X), sample epsilon = N(0,I),
        then  z = z_mean + sqrt(var) * epsilon
        """
        z_mu, z_log_var = args

        batch = K.shape(z_mu)[0]
        dim = K.int_shape(z_mu)[1]

        epsilon = K.random_normal(shape=(batch, dim))
        return z_mu + K.exp(0.5 * z_log_var) * epsilon

    def encoder_net(self, inputs, input_shape_box, conv_params, latent_dim):
        initialiser = keras.initializers.he_uniform()
        convlayers = layers.ConvLayers(conv_params, initialiser)

        x = convlayers.conv_layers(inputs, input_shape_box)
        x = Flatten(data_format=self.data_format)(x)

        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decoder_net(self, latent_inputs, fcc_params):
        initialiser = keras.initializers.he_uniform()
        denselayers = layers.FCCLayers(fcc_params, initialiser)

        x = denselayers.fcc_layers(latent_inputs)
        outputs = Dense(1, activation='linear')(x)
        return outputs

    def predict_latent_mean_std(self, testing_data):
        encoder = self.encoder
        z_mean, z_var, z = encoder.predict(testing_data)
        return z_mean, np.exp(0.5 * z_var)

