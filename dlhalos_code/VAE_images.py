from tensorflow.keras.layers import Lambda, Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np
import time


class VAE:
    def __init__(self, training_data, validation_data, use_multiprocessing=False, workers=1, verbose=1,
                 inter_dim=512, latent_dim=2, batch_size=128, epochs=10, beta=1, plot_models=True):

        # self.training_generator = training_generator
        # self.validation_generator = validation_generator
        #
        # self.input_shape = training_generator.dim
        self.input_shape = training_data.shape[1]
        self.beta = beta
        self.num_epochs = epochs
        self.use_multiprocessing = use_multiprocessing
        self.workers = workers
        self.verbose = verbose

        input_encoder = Input(shape=(self.input_shape,), name='encoder_input')
        z_mean, z_log_var, z = self.encoder_net(input_encoder, inter_dim, latent_dim)
        encoder = Model(input_encoder, [z_mean, z_log_var, z], name='encoder')
        print(encoder.summary())
        if plot_models is True:
            plot_model(encoder, to_file='my_encoder.png', show_shapes=True)

        input_decoder = Input(shape=(latent_dim,), name='decoder_input')
        output_decoder = self.decoder_net(input_decoder, inter_dim, self.input_shape)
        decoder = Model(input_decoder, output_decoder, name='decoder')
        print(decoder.summary())
        if plot_models is True:
            plot_model(decoder, to_file='my_decoder.png', show_shapes=True)

        vae_output = decoder(encoder(input_encoder)[2])
        vae = Model(input_encoder, vae_output, name='vae')

        vae_loss = K.mean(self.reconstructuion_loss(input_encoder, vae_output) + self.KL_loss(z_mean, z_log_var))
        vae.add_loss(vae_loss)

        vae.compile(optimizer='adam')
        print(vae.summary())
        if plot_models is True:
            plot_model(vae, to_file='my_vae.png', show_shapes=True)

        t0 = time.time()
        # history = vae.fit_generator(generator=self.training_generator, validation_data=self.validation_generator,
        #                               use_multiprocessing=self.use_multiprocessing, workers=self.workers,
        #                               verbose=self.verbose, epochs=self.num_epochs, shuffle=True)
        history = vae.fit(training_data, epochs=epochs, batch_size=batch_size,
                          validation_data=(validation_data, None))
        t1 = time.time()
        print("This model took " + str((t1 - t0)/60) + " minutes to train.")

        self.vae = vae
        self.history = history
        self.models = (encoder, decoder)

    @staticmethod
    def reconstructuion_loss(inputs, outputs):
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= original_dim
        return reconstruction_loss

    @staticmethod
    def KL_loss(z_mu, log_z_variance):
        kl_loss = 1 + log_z_variance - K.square(z_mu) - K.exp(log_z_variance)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss

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

    def encoder_net(self, inputs, intermediate_dim, latent_dim):
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decoder_net(self, latent_inputs, intermediate_dimensions, input_dimensions):
        y = Dense(intermediate_dimensions, activation='relu')(latent_inputs)
        outputs = Dense(input_dimensions, activation='sigmoid')(y)
        return outputs

    def predict_latent_mean_std(self, testing_data):
        encoder = self.models[0]
        z_mean, z_var, z = encoder.predict(testing_data)
        return z_mean, np.exp(0.5 * z_var)

    def generate_new_image(self, testing_data):
        new_image = self.vae.predict(testing_data)
        return new_image

if __name__ == "__main__":

    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    model = VAE(x_train, x_test[:10], epochs=10)
