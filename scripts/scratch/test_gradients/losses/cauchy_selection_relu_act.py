import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras as keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
from pickle import load
import tensorflow.keras.backend as K
import numpy as np


def predict_from_model(model, epoch, gen_train, gen_val, training_IDs, training_labels_IDS, val_IDs, val_labels_IDs,
                       scaler, path_model):
    pred = model.predict(gen_train, verbose=1, workers=0)
    truth_rescaled = np.array([training_labels_IDS[ID] for ID in training_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(path_model + "predicted_training_"+ epoch + ".npy", h_m_pred)
    np.save(path_model + "true_training_"+ epoch + ".npy", true)
    pred = model.predict(gen_val, verbose=1, workers=0)
    truth_rescaled = np.array([val_labels_IDs[ID] for ID in val_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(path_model + "predicted_val_"+ epoch + ".npy", h_m_pred)
    np.save(path_model + "true_val_" + epoch + ".npy", true)


def double_relu_activation(inputs):
    theta = K.cast_to_floatx(1)
    abs_inputs = K.abs(inputs)
    return inputs * K.cast(K.less_equal(abs_inputs, theta), K.floatx())


def clip_activation(inputs):
    upper_bound = K.cast_to_floatx(1)
    lower_bound = K.cast_to_floatx(-1)
    return K.maximum(lower_bound, K.minimum(inputs, upper_bound))


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # Load data

    path_data = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/"

    scaler_output = load(open(path_data + 'scaler_output.pkl', "rb"))
    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params_tr)

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)

    ######### TRAIN THE MODEL ################

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
                 "/cauchy_selec_relu_last_act/"

    # Define model

    kernel_reg = regularizers.l2(0.0005)
    bias_reg = regularizers.l2(0.0005)
    activation = "linear"
    relu = True

    params_all_conv = {'activation': activation, 'relu': relu, 'strides': 1, 'padding': 'same',
                       'bn': False, 'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    params_all_fcc = {'bn': False, 'dropout': 0.4, 'activation': activation, 'relu': relu,
                      'kernel_regularizer': kernel_reg}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc},
                 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {'activation': double_relu_activation}
                 }

    training = True
    testing = False

    if training:
        print("Training mode")

        # callbacks
        filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
        checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5, save_weights_only=True)
        csv_logger = CSVLogger(path_model + "/training.log", separator=',')
        lrate = callbacks.LearningRateScheduler(CNN.lr_scheduler)
        callbacks_list = [checkpoint_call, csv_logger, lrate]

        lr = 0.0001
        Model = CNN.CNN(param_conv, param_fcc, model_type="regression", train=False, compile=True,
                        training_generator=generator_training, steps_per_epoch=1000,
                        validation_generator=generator_validation, validation_steps=len(generator_validation),
                        lr=0.0001,
                        # callbacks=callbacks_list,
                        metrics=['mae', 'mse'],
                        num_epochs=10, dim=generator_training.dim,
                        loss=lf.cauchy_selection_loss(),
                        max_queue_size=10, use_multiprocessing=False, workers=0, verbose=1,
                        num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1)

        m = Model.model

        trained_weights = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin" \
                          "/larger_net/lr_decay/mse/model/weights.10.hdf5"
        m.load_weights(trained_weights)

        m.fit_generator(generator=generator_training, steps_per_epoch=len(generator_training),
                        use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10,
                        callbacks=callbacks_list, shuffle=True,
                        epochs=100, initial_epoch=10,
                        validation_data=generator_validation, validation_steps=len(generator_validation))

        # m = Model.model
        # m.load_weights(trained_weights)
        # predictions = CNN.CauchyLayer()(m.layers[-1].output)
        # new_model = keras.Model(inputs=m.input, outputs=predictions)
        #
        # optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        # loss_c = lf.cauchy_selection_loss_trainable_gamma(new_model.layers[-1])
        # # lr_metric = get_lr_metric(optimiser)
        #
        # new_model.compile(loss=loss_c, optimizer=optimiser, metrics=['mae', 'mse'])
        # new_model.save_weights(path_model + 'model/initial_weights.h5')
        #
        # new_model.fit_generator(generator=generator_training, steps_per_epoch=len(generator_training),
        #                         use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10,
        #                         callbacks=callbacks_list, shuffle=True, epochs=100,
        #                         validation_data=generator_validation,
        #                         validation_steps=len(generator_validation)
        #                         )

    if testing:
        lr = 0.0001
        Model = CNN.CNN(param_conv, param_fcc, model_type="regression", train=False, compile=False,
                        initial_epoch=10,
                        training_generator=generator_training,
                        lr=0.0001, metrics=['mae', 'mse'],
                        num_epochs=11, dim=generator_training.dim,
                        loss=lf.cauchy_selection_loss(),
                        max_queue_size=10, use_multiprocessing=False, workers=0, verbose=1,
                        num_gpu=1, save_summary=True, path_summary=path_model, validation_freq=1)

        m = Model.model
        predictions = CNN.CauchyLayer()(m.layers[-1].output)
        new_model = keras.Model(inputs=m.input, outputs=predictions)

        optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        loss_c = lf.cauchy_selection_loss_trainable_gamma(new_model.layers[-1])

        epochs = ["20", "30", "50"]

        for epoch in epochs:
            new_model.load_weights(path_model + "/model/weights." + epoch + ".hdf5")
            predict_from_model(new_model, epoch, generator_training, generator_validation,
                           training_particle_IDs, training_labels_particle_IDS,
                           val_particle_IDs, val_labels_particle_IDS, scaler_output, path_model)

