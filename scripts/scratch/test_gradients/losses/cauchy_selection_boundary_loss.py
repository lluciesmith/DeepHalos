import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
from pickle import dump, load
import numpy as np

if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    # Load data

    path_data = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/"

    scaler_output = load(open(path_data + 'scaler_output.pkl', "rb"))

    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))

    # Create the generators for training

    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=False, **params_tr)

    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)

    ######### TRAINING MODEL FROM MSE TRAINED ONE ##############

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
                 "/cauchy_selec_bound_test/"

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
    csv_logger = CSVLogger(path_model + "/training.log", separator=',')
    lrate = callbacks.LearningRateScheduler(CNN.lr_scheduler)
    callbacks_list = [checkpoint_call, csv_logger, lrate]

    lr = 0.0001
    Model = CNN.CNN(param_conv, param_fcc, model_type="regression", train=False, compile=True,
                    initial_epoch=10,
                    training_generator=generator_training, dim=generator_training.dim,
                    # validation_generator=generator_validation, validation_steps=len(generator_validation),
                    lr=0.0001,
                    callbacks=callbacks_list,
                    metrics=['mae', 'mse'],
                    num_epochs=11,
                    loss=lf.cauchy_selection_loss_fixed_boundary(),
                    max_queue_size=1, use_multiprocessing=False, workers=0, verbose=1,
                    num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1)

    m = Model.model
    m.load_weights(trained_weights)
    m.fit_generator(generator=generator_training, steps_per_epoch=len(generator_training),
                    use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10,
                    callbacks=callbacks_list, shuffle=True,
                    epochs=20, initial_epoch=11,
                    validation_data=generator_validation, validation_steps=50)

# import time
# import gc
#
# epochs = 2
#
#
# t00 = time.time()
# for epoch in range(epochs):
#     t0 = time.time()
#     epoch_init = epoch
#     h = m.fit_generator(generator=generator_training, steps_per_epoch=training_steps,
#                     use_multiprocessing=False, workers=1, verbose=1, max_queue_size=1,
#                     # callbacks=callbacks_list,
#                     initial_epoch=epoch_init, epochs=epoch_init+1)
#     t1 = time.time()
#     print("Epoch " + str(epoch) + " took " + str((t1 - t0) / 60) + " minutes.")
#     print("Evaluate generator for epoch " + str(epoch))
#     batches = len(generator_validation)
#     for i in range(batches):
#         x, y = generator_validation[i]
#         h1 = m.evaluate(x, y, reset_metrics=False)
#         print("Done evaluation on batch " + str(i) + " of epoch " + str(epoch))
# t11 = time.time()
# print("Training two epochs took " + str((t11 - t00) / 60) + " minutes.")
#
# l = m.evaluate_generator(generator=generator_validation, verbose=1, use_multiprocessing=False, max_queue_size=1,
#                          workers=1, steps=val_steps)
#     print("Done evaluation")
# t11 = time.time()
# print("Training two epochs took " + str((t11 - t00) / 60) + " minutes.")
#
#
# t00 = time.time()
# for epoch in range(epochs):
#     t0 = time.time()
#     batches = len(generator_training)
#     for i in range(batches):
#         x, y = generator_training[i]
#         h1 = m.train_on_batch(x, y, reset_metrics=False)
#         print("Done batch " + str(i) + " of epoch " + str(epoch))
#     gc.collect()
#     t1 = time.time()
#     print("Epoch " + str(epoch) + " took " + str((t1 - t0) / 60) + " minutes.")
#     print("Evaluate generator for epoch " + str(epoch))
#     l1 = m.evaluate_generator(generator=generator_validation, verbose=1, use_multiprocessing=False)
#     print("Done evaluation")
# t11 = time.time()
# print("Training two epochs took " + str((t11 - t00) / 60) + " minutes.")
#
#

