import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
from pickle import dump, load
import tensorflow.keras as keras
import numpy as np


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def lr_scheduler(epoch):
    init_lr = 0.0001
    n = 10
    if epoch < n:
        return init_lr
    else:
        return init_lr * np.math.exp(0.05 * (n - epoch))


def lr_schefuler_half(epoch):
    init_lr = 0.0001
    if epoch < 10:
        return init_lr
    else:
        drop_rate = 0.5
        epoch_drop = 10
        return init_lr * drop_rate**np.floor(epoch / epoch_drop)


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
                                          shuffle=False, **params_tr)

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)


    ######### TRAIN THE MODEL ################

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
                 "/cauchy_selec_bound_gamma08/"

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
                 'last': {}
                 }

    # callbacks
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=1, save_weights_only=True)
    lrate = callbacks.LearningRateScheduler(lr_schefuler_half)
    cbk = CNN.CollectWeightCallback(layer_index=-1)
    csv_logger = CSVLogger(path_model + "/training.log", separator=',')
    callbacks_list = [checkpoint_call, csv_logger, lrate, cbk]

    lr = 0.0001
    Model = CNN.CNN(param_conv, param_fcc, model_type="regression", train=False, compile=False,
                    initial_epoch=10,
                    training_generator=generator_training,
                    lr=0.0001, callbacks=callbacks_list, metrics=['mae', 'mse'],
                    num_epochs=11, dim=generator_training.dim,
                    loss=lf.cauchy_selection_loss(),
                    max_queue_size=10, use_multiprocessing=False, workers=1, verbose=1,
                    num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1)

    m = Model.model
    trained_weights = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin" \
                      "/larger_net/lr_decay/mse/model/weights.10.hdf5"
    m.load_weights(trained_weights)
    predictions = CNN.CauchyLayer(init_value=0.8)(m.layers[-1].output)
    new_model = keras.Model(inputs=m.input, outputs=predictions)

    optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
    loss_c = lf.cauchy_selection_loss_fixed_boundary_trainable_gamma(new_model.layers[-1])
    lr_metric = get_lr_metric(optimiser)

    new_model.compile(loss=loss_c, optimizer=optimiser, metrics=['mae', 'mse', lr_metric])
    new_model.save_weights(path_model + 'model/initial_weights.h5')

    new_model.fit_generator(generator=generator_training, steps_per_epoch=len(generator_training),
                            use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10,
                            callbacks=callbacks_list, shuffle=True, epochs=30, initial_epoch=10,
                            validation_data=generator_validation,
                            validation_steps=len(generator_validation)
                            )
    print(cbk.weights)
    np.save(path_model + 'gamma.npy', cbk.weights)
