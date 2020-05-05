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


def predict_from_model(model, epoch, gen_train, gen_val, training_IDs, training_labels_IDS, val_IDs, val_labels_IDs,
                       scaler, path_model):
    pred = model.predict_generator(gen_train, use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10)
    truth_rescaled = np.array([training_labels_IDS[ID] for ID in training_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(path_model + "predicted_training_"+ epoch + ".npy", h_m_pred)
    np.save(path_model + "true_training_"+ epoch + ".npy", true)
    pred = model.predict_generator(gen_val, use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10)
    truth_rescaled = np.array([val_labels_IDs[ID] for ID in val_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(path_model + "predicted_val_"+ epoch + ".npy", h_m_pred)
    np.save(path_model + "true_val_" + epoch + ".npy", true)

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
                 "/cauchy_selec_bound_gamma_l1_reg/"

    # Define model

    kernel_reg = regularizers.l1(0.0005)
    bias_reg = regularizers.l1(0.0005)
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

    # Train for one epoch using MSE loss

    lr = 0.0001
    Model = CNN.CNN(param_conv, param_fcc, model_type="regression",
                    training_generator=generator_training,
                    lr=lr, num_epochs=1, dim=generator_validation.dim, loss='mse',
                    max_queue_size=10, use_multiprocessing=False, workers=0, verbose=1,
                    num_gpu=1, save_summary=True, path_summary=path_model, validation_freq=1, train=True)

    # Define new model

    m = Model.model
    predictions = CNN.CauchyLayer()(m.layers[-1].output)
    new_model = keras.Model(inputs=m.input, outputs=predictions)

    optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
    loss_c = lf.cauchy_selection_loss_fixed_boundary_trainable_gamma(new_model.layers[-1])

    new_model.compile(loss=loss_c, optimizer=optimiser)

    ######### training/testing #########

    training = True
    testing = False

    if training:

        # callbacks
        filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
        checkpoint_call = callbacks.ModelCheckpoint(filepath, period=1, save_weights_only=True)
        lrate = callbacks.LearningRateScheduler(lr_schefuler_half)
        cbk = CNN.CollectWeightCallback(layer_index=-1)
        csv_logger = CSVLogger(path_model + "/training.log", separator=',')
        callbacks_list = [checkpoint_call, csv_logger, lrate, cbk]

        new_model.save_weights(path_model + 'model/initial_weights.h5')

        new_model.fit_generator(generator=generator_training, steps_per_epoch=len(generator_training),
                                use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10,
                                callbacks=callbacks_list, shuffle=True, epochs=40, initial_epoch=1,
                                validation_data=generator_validation,
                                validation_steps=len(generator_validation)
                                )
        print(cbk.weights)
        np.save(path_model + 'gamma.npy', cbk.weights)

    if testing:

        epoch = '19'
        new_model.load_weights(path_model + 'model/weights.' + epoch + '.hdf5')
        predict_from_model(new_model, epoch, generator_training, generator_validation,
                           training_particle_IDs, training_labels_particle_IDS,
                           val_particle_IDs,  val_labels_particle_IDS,
                           scaler_output, path_model)

        # Also make predictions for a larger validation set

        larger_val_particle_IDs = load(open(path_data + 'larger_validation_set.pkl', 'rb'))
        larger_val_labels_particle_IDS = load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))

        generator_larger_validation = tn.DataGenerator(larger_val_particle_IDs, larger_val_labels_particle_IDS,
                                                       s.sims_dic, shuffle=False, **params_val)
        pred = new_model.predict_generator(generator_larger_validation, use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10)
        truth_rescaled = np.array([larger_val_labels_particle_IDS[ID] for ID in larger_val_particle_IDs])
        h_m_pred = scaler_output.inverse_transform(pred.reshape(-1, 1)).flatten()
        true = scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
        np.save(path_model + "predicted_larger_val_"+ epoch + ".npy", h_m_pred)
        np.save(path_model + "true_larger_val_" + epoch + ".npy", true)
