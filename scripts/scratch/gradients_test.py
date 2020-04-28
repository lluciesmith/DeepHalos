import sys
import tensorflow as tf
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
from pickle import dump, load
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model


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

        ######### TRAINING MODEL FROM MSE TRAINED ONE ##############

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
                 "/cauchy_selec_bound/"

    # Load weights

    trained_weights = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin" \
                      "/larger_net/lr_decay/mse/model/weights.10.hdf5"

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
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
    lrate = callbacks.LearningRateScheduler(CNN.lr_scheduler)
    callbacks_list = [checkpoint_call, csv_logger, lrate]

    lr = 0.0001
    Model = CNN.CNN(param_conv, param_fcc, model_type="regression", train=False, compile=True,
                    initial_epoch=10,
                    training_generator=generator_training, dim=generator_training.dim,
                    # validation_generator=generator_validation, validation_steps=len(generator_validation),
                    lr=0.0001,
                    # callbacks=callbacks_list,
                    metrics=['mae', 'mse'],
                    num_epochs=11,
                    loss=lf.cauchy_selection_loss_fixed_boundary(),
                    max_queue_size=1, use_multiprocessing=False, workers=0, verbose=1,
                    num_gpu=1, save_summary=True, path_summary=path_model, validation_freq=1)

    m = Model.model
    m.load_weights(trained_weights)

    # Get gradients

    x, y = generator_training[0]
    X = tf.convert_to_tensor(x, dtype="float32")
    Y = tf.convert_to_tensor(y.reshape(100, 1), dtype="float32")
    l = lf.cauchy_selection_loss_fixed_boundary()


    with tf.compat.v1.Session() as ses:
        init_op = tf.global_variables_initializer()
        ses.run(init_op)
        predictions = Model.model(X)
        # loss_value = mean_squared_error(Y, predictions)
        loss_value = l(Y, predictions)
        grads = tf.gradients(loss_value, [predictions])
        p = ses.run(predictions)
        ls = ses.run(loss_value)
        g = ses.run(grads)



    # On local machine, run:

    import matplotlib.pyplot as plt

    L = lf.ConditionalCauchySelectionLoss()
    d = {}
    x = {}

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    axes[0].scatter(p, d, c=(L.loss(d.reshape(100,1), p) - ls)/ls * 100, vmin=-0.01, vmax=0.01)
    axes[0].set_title(r"$\mathcal{L}_B (x, d)$")

    CS1 = axes[1].scatter(p, d, c=(L.dloss(d.reshape(100,1), p) - g)/g * 100, vmin=-0.01, vmax=0.01)
    cbar = f.colorbar(CS1)
    cbar.ax.set_ylabel(r'(Analytic - tensorflow)/tensorflow $\times 100$')
    axes[1].set_title(r"$\partial \mathcal{L}_B (x, d)/\partial x$")

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("d")
    axes[1].set_xlabel("x")
    plt.subplots_adjust(left=0.1, bottom=0.14, top=0.9, wspace=0)
