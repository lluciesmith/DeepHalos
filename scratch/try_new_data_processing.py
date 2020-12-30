import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow
from tensorflow.keras.models import load_model
import dlhalos_code.data_processing as tn
import time
import numpy as np


########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########


# First you will have to load the simulation

all_sims = ["0", "1", "2", "3", "4", "5"]
s = tn.SimulationPreparation(all_sims)

training_sims = ["0", "2", "3", "4", "5"]
validation_sims = ["1"]
batch_size = 80
rescale_mean = 1.005
rescale_std = 0.05050

training_set = tn.InputsPreparation(training_sims, load_ids=True)
generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
                                      batch_size=batch_size, rescale_mean=rescale_mean, rescale_std=rescale_std)

validation_set = tn.InputsPreparation(validation_sims, load_ids=True, random_subset_each_sim=4000,
                                      scaler_output=training_set.scaler_output)
generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
                                        batch_size=batch_size, rescale_mean=rescale_mean, rescale_std=rescale_std)

######### TRAINING MODEL ##############

path_model = "/lfstev/deepskies/luisals/scratch2/"

# checkpoint
filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

# save histories
csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=False)

callbacks_list = [checkpoint_call, csv_logger]
tensorflow.compat.v1.set_random_seed(7)

param_conv = {'conv_1': {'num_kernels': 4, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
              'conv_2': {'num_kernels': 8, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
              'conv_3': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
              'conv_4': {'num_kernels': 4, 'dim_kernel': (1, 1, 1),
                         'strides': 1, 'padding': 'same', 'pool': None, 'bn': True}
              }

param_fcc = {  # 'dense_1': {'neurons': 1024, 'bn': False, 'dropout': 0.2},
    'dense_1': {'neurons': 256, 'bn': True, 'dropout': 0.4},
    'dense_2': {'neurons': 128, 'bn': False, 'dropout': 0.4}}

Model = CNN.CNN(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                validation_generator=generator_validation, callbacks=callbacks_list, num_epochs=100, dim=(51, 51, 51),
                max_queue_size=10, use_multiprocessing=True, workers=2, verbose=1, lr=0.001, validation_freq=1,
                train=True)


np.save(path_model + "/history_100_epochs_mixed_sims.npy", Model.history)
Model.model.save(path_model + "/model_100_epochs_mixed_sims.h5")


########## OPTION 2 ######################

validation_set = tn.InputsPreparation(validation_sims, load_ids=True, scaler_output=training_set.scaler_output)
generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
                                        batch_size=batch_size, rescale_mean=rescale_mean, rescale_std=rescale_std)

Model.model.fit_generator()

# training_set = tn.InputsPreparation(training_sims, load_ids=True)
# generator_training2 = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
#                                       batch_size=100000, rescale_mean=rescale_mean, rescale_std=rescale_std)
# X_train, y_train = generator_training2[0]
#
# validation_set = tn.InputsPreparation(validation_sims, load_ids=True, scaler_output=training_set.labels_scaler,
#                                       random_subset_each_sim=4000)
# generator_validation2 = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
#                                         batch_size=4000, rescale_mean=rescale_mean, rescale_std=rescale_std)
# X_val, y_val = generator_validation2[0]
#
# Model2 = CNN.CNN(param_conv, param_fcc, dim=(51, 51, 51),
#                 training_generator=generator_training, validation_generator=None, validation_freq=1,
#                 callbacks=callbacks_list, use_multiprocessing=True, num_epochs=100,
#                 workers=12, verbose=1, model_type="regression", lr=0.0001, train=False)
#
# history = Model2.model.fit(X_train, y_train, batch_size=80, verbose=1, epochs=100, validation_data=(X_val, y_val),
#                           shuffle=True, callbacks=callbacks_list)
#
#
#
# ####### tests
#
# s = tn.SimulationPreparation(["0", "1"])
#
# def run():
#     t0 = time.time()
#     training_sims = ["0", "1"]
#     training_set = tn.InputsPreparation(training_sims, load_ids=True)
#     generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
#                                          batch_size=10000, rescale_mean=0, rescale_std=1)
#     x, y = generator_training[0]
#     t1 = time.time()
#     print("Loading 10000 particles took " + str((t1 - t0) / 60))
#
# run()
#
#
# def run2():
#     all_sims = ["0", "1", "2", "4", "5", "6"]
#     s = tn.SimulationPreparation(all_sims)
#
#     t0 = time.time()
#     training_sims = ["0", "2", "4", "5", "6"]
#     training_set = tn.InputsPreparation(training_sims, load_ids=True)
#     generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
#                                           batch_size=100000, rescale_mean=0, rescale_std=1)
#     x, y = generator_training[0]
#     t1 = time.time()
#     print("Loading 100000 particles took " + str((t1 - t0) / 60))
#
#
# run2()


