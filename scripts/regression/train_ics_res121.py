import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
import tensorflow
import dlhalos_code.data_processing as tn
import numpy as np
from pickle import dump


########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

path_model = "/lfstev/deepskies/luisals/regression/ics_res121/stride1/"

# First you will have to load the simulation

all_sims = ["0", "1", "2", "3", "4", "5"]
s = tn.SimulationPreparation(all_sims)

training_sims = ["0", "2", "3", "4", "5"]
validation_sims = ["1"]
batch_size = 40
rescale_mean = 1.005
rescale_std = 0.05050
dim = (121, 121, 121)

training_set = tn.InputsPreparation(training_sims, load_ids=True)
generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
                                      batch_size=batch_size, rescale_mean=rescale_mean, rescale_std=rescale_std,
                                      dim=dim)

validation_set = tn.InputsPreparation(validation_sims, load_ids=True, random_subset_each_sim=4000,
                                      scaler_output=training_set.scaler_output)
generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
                                        batch_size=batch_size, rescale_mean=rescale_mean, rescale_std=rescale_std,
                                        dim=dim)

dump(training_set.scaler_output, open(path_model + 'scaler_output.pkl', 'wb'))

######### TRAINING MODEL ##############

# checkpoint
filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

# save histories
csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=False)

callbacks_list = [checkpoint_call, csv_logger]
tensorflow.compat.v1.set_random_seed(7)

param_conv = {'conv_1': {'num_kernels': 4, 'dim_kernel': (3, 3, 3), 'pool_size':(2, 2, 2),
                         'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
              'conv_2': {'num_kernels': 8, 'dim_kernel': (3, 3, 3), 'pool_size':(2, 2, 2),
                         'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
              'conv_3': {'num_kernels': 16, 'dim_kernel': (3, 3, 3), 'pool_size':(2, 2, 2),
                         'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
              'conv_4': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool_size':(2, 2, 2),
                         'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
              'conv_5': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool_size':(2, 2, 2),
                         'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
              'conv_6': {'num_kernels': 16, 'dim_kernel': (1, 1, 1), 'pool_size':(3, 3, 3),
                          'strides': 1, 'padding': 'same', 'pool': None, 'bn': True}
              }

param_fcc = {  # 'dense_1': {'neurons': 1024, 'bn': False, 'dropout': 0.2},
    'dense_1': {'neurons': 256, 'bn': True, 'dropout': 0.4},
    'dense_2': {'neurons': 128, 'bn': False, 'dropout': 0.4}}

Model = CNN.CNN(param_conv, param_fcc, dim=dim,
                training_generator=generator_training, validation_generator=generator_validation, validation_freq=1,
                callbacks=callbacks_list, num_epochs=100,
                use_multiprocessing=True, workers=2, max_queue_size=10,
                verbose=1, model_type="regression", lr=0.0001, train=False)


np.save(path_model + "/history_100_epochs_mixed_sims.npy", Model.history)
Model.model.save(path_model + "/model_100_epochs_mixed_sims.h5")