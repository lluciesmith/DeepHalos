import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow
import dlhalos_code.data_processing as tn


########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########


# First you will have to load the simulation

all_sims = ["0", "1", "2", "4", "5", "6"]
s = tn.SimulationPreparation(all_sims)

training_sims = ["0", "2", "4", "5", "6"]
validation_sims = ["1"]
batch_size = 80
rescale_mean = 1.005
rescale_std = 0.0505

training_set = tn.InputsPreparation(training_sims, load_ids=True)
generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
                                      batch_size=batch_size, rescale_mean=0, rescale_std=1)

validation_set = tn.InputsPreparation(validation_sims, load_ids=True, scaler_output=training_set.labels_scaler,
                                      random_subset_each_sim=4000)
generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
                                        batch_size=batch_size, rescale_mean=rescale_mean, rescale_std=rescale_std)

######### TRAINING MODEL ##############

path_model = "/lfstev/deepskies/luisals/scratch/"

# checkpoint
filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

# save histories
csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=False)

# decay the learning rate
lr_decay = LearningRateScheduler(CNN.lr_scheduler)

# callbacks_list = [checkpoint_call, csv_logger]
callbacks_list = [checkpoint_call, csv_logger, lr_decay]

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

Model = CNN.CNN(param_conv, param_fcc, dim=(51, 51, 51),
                training_generator=generator_training, validation_generator=None, validation_freq=1,
                callbacks=callbacks_list, use_multiprocessing=True, num_epochs=100,
                workers=12, verbose=1, model_type="regression", lr=0.0001, train=True)



