import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
import tensorflow as tf
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
import tensorflow.compat.v1
from utilss.old import generators_training as gbc
import time

if __name__ == "__main__":
    ########### CREATE GENERATORS FOR SIMULATIONS #########

    path_model = "/lfstev/deepskies/luisals/regression/test_spherical_overden/standardize/scale_39/"
    ph = "/lfstev/deepskies/luisals/"
    index_scale = 39

    rescale_mean = 1.004
    rescale_std = 0.05

    t0 = time.time()

    with tf.device('/cpu:0'):
        f = "random_training_set.txt"

        ids_0, so_0 = gbc.get_ids_and_SO_labels(sim="0", ids_filename=f, fitted_scaler=None, path=ph, index=index_scale)
        ids_3, so_3 = gbc.get_ids_and_SO_labels(sim="3", ids_filename=f, fitted_scaler=None, path=ph, index=index_scale)
        ids_4, so_4 = gbc.get_ids_and_SO_labels(sim="4", ids_filename=f, fitted_scaler=None, path=ph, index=index_scale)
        ids_5, so_5 = gbc.get_ids_and_SO_labels(sim="5", ids_filename=f, fitted_scaler=None, path=ph, index=index_scale)

        ran = np.random.choice(np.arange(20000), 10000)
        ran_val = np.random.choice(np.arange(20000), 4000)
        np.save(path_model + "ran.npy", ran)
        np.save(path_model + "ran_val1.npy", ran_val)

        sims = ["0", "3", "4", "5"]
        ids_s = [ids_0[ran], ids_3[ran], ids_4[ran], ids_5[ran]]
        output_ids, output_scaler = gbc.get_standard_scaler_and_transform([so_0[ran], so_3[ran], so_4[ran], so_5[ran]])

        generator_training = gbc.create_generator_multiple_sims(sims, ids_s, output_ids, batch_size=80,
                                                                rescale_mean=rescale_mean, rescale_std=rescale_std)

        ids_1, so_1 = gbc.get_ids_and_SO_labels(sim="1", ids_filename=f,
                                            fitted_scaler=output_scaler, path=ph, index=index_scale)
        generator_1 = gbc.create_generator_sim(ids_1[ran_val], so_1[ran_val], batch_size=80,
                                               rescale_mean=rescale_mean, rescale_std=rescale_std,
                                               path=ph + "reseed1_simulation/training_set/")

        # ids_2, so_2 = get_ids_and_SO_labels(sim="2", ids_filename=f,
        #                                     fitted_scaler=output_scaler, path=ph, index=index_scale)
        # generator_2 = gbc.create_generator_sim(ids_2, so_2, batch_size=80,
        #                                        rescale_mean = rescale_mean, rescale_std = rescale_std,
        #                                        path=ph + "reseed2_simulation/training_set/")

    t1 = time.time()
    print("Loading generators took " + str((t1 - t0) / 60) + " minutes.")

    ######### TRAINING MODEL ##############

    with tf.device('/gpu:0'):

        # checkpoint
        filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
        checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

        # save histories
        csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=False)

        # decay the learning rate
        # lr_decay = LearningRateScheduler(CNN.lr_scheduler)

        callbacks_list = [checkpoint_call, csv_logger]
        # callbacks_list = [checkpoint_call, csv_logger, lr_decay]

        tensorflow.compat.v1.set_random_seed(7)
        # param_conv = {'conv_1': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
        #                          'strides': 1, 'padding': 'valid',
        #                          'pool': True, 'bn': False},  # 24x24x24
        #               'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3),
        #                          'strides': 1, 'padding': 'valid',
        #                          'pool': True, 'bn': False}, # 11x11x11
        #               'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3),
        #                          'strides': 1, 'padding': 'valid',
        #                          'pool': True, 'bn': False}, # 9x9x9
        #               'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3),
        #                          'strides': 1, 'padding': 'valid',
        #                          'pool': False, 'bn': False}, # 7x7x7
        #               }
        # param_fcc = {'dense_1': {'neurons': 256, 'dropout': 0.2},
        #              'dense_2': {'neurons': 128, 'dropout': 0.2}}

        param_conv = {'conv_1': {'num_kernels': 8, 'dim_kernel': (3, 3, 3),
                                 'strides': 2, 'padding': 'valid',
                                 'pool': True, 'bn': False},
                      'conv_2': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
                                 'strides': 1, 'padding': 'valid',
                                 'pool': True, 'bn': False},
                      'conv_3': {'num_kernels': 32, 'dim_kernel': (3, 3, 3),
                                 'strides': 1, 'padding': 'valid',
                                 'pool': False, 'bn': False},
                      }
        param_fcc = {'dense_1': {'neurons': 256, 'bn': False
                                 # 'dropout': 0.1
                                 },
                     'dense_2': {'neurons': 128, 'bn': False
                                 # 'dropout': 0.1
                                 }
                     }

        Model = CNN.CNN(generator_training, param_conv, model_type="regression", validation_generator=generator_1,
                        callbacks=callbacks_list, num_epochs=80, use_multiprocessing=True, workers=14, verbose=1,
                        lr=0.0001)

        model = Model.model
        history = Model.history

        np.save(path_model + "/history_80_epochs_mixed_sims.npy", history.history)
        model.save(path_model + "/model_80_epochs_mixed_sims.h5")



# def normalise_distribution_to_given_variance2(samples, variance):
#     mean_samples = np.mean(samples)
#     std_samples = np.std(samples)
#     samples_0mean_1var = (samples - mean_samples)/std_samples
#     s2 = (samples_0mean_1var - np.mean(samples_0mean_1var)) * np.sqrt(variance)
#     samples_var0 = (s2 - np.mean(s2)) + mean_samples
#     np.testing.assert_allclose(np.mean(samples_var0), mean_samples,
#                                err_msg="Mean of new samples is " + str(np.mean(samples_var0)))
#     np.testing.assert_allclose(np.var(samples_var0), variance,
#                                err_msg="Mean of new samples is " + str(np.var(samples_var0)))
#     return samples_var0