import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from utils import generators_training as gbc
import time
import tensorflow



########### CREATE GENERATORS FOR SIMULATIONS #########

# ph = "share/hypatia/lls/deep_halos/"
path_model = "/lfstev/deepskies/luisals/regression/train_mixed_sims/fit/"
ph = "/lfstev/deepskies/luisals/"

rescale_mean = 1.004
rescale_std = 0.05

t0 = time.time()

f = "random_training_set.txt"
ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=None)
ids_3, mass_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename=f, fitted_scaler=None)
ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename=f, fitted_scaler=None)
ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename=f, fitted_scaler=None)

# training set
sims = ["0", "3", "4", "5"]
ids_s = [ids_0, ids_3, ids_4, ids_5]
mass_ids = [mass_0, mass_3, mass_4, mass_5]
output_ids, output_scaler = gbc.get_standard_scaler_and_transform([mass_0, mass_3, mass_4, mass_5])
generator_training = gbc.create_generator_multiple_sims(sims, ids_s, output_ids, batch_size=80000,
                                                        rescale_mean=rescale_mean, rescale_std=rescale_std)
X, y = generator_training[0]

# validation set
ran_val = np.random.choice(np.arange(20000), 4000)
np.save(path_model + "ran_val1.npy", ran_val)
ids_1, mass_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename=f, fitted_scaler=output_scaler)
generator_1 = gbc.create_generator_sim(ids_1[ran_val], mass_1[ran_val], batch_size=4000, rescale_mean=rescale_mean,
                                       rescale_std=rescale_std, path=ph + "reseed1_simulation/training_set/")
X_val1, y_val1 = generator_1[0]

ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=output_scaler)
generator_0 = gbc.create_generator_sim(ids_0, mass_0, batch_size=20000, rescale_mean=rescale_mean,
                                       rescale_std=rescale_std, path=ph + "training_simulation/training_set/")
X_val0, y_val0 = generator_0[0]

# ids_2, mass_2 = gbc.get_ids_and_regression_labels(sim="2", ids_filename=f, fitted_scaler=output_scaler)
# generator_2 = gbc.create_generator_sim(ids_2[ran_val], mass_2[ran_val], batch_size=80,
#                                        path=ph + "reseed2_simulation/training_set/")

t1 = time.time()
print("Loading generators took " + str((t1 - t0) / 60) + " minutes.")

######### TRAINING MODEL ##############

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
                         'strides': 1, 'padding': 'same', 'pool': True, 'bn': True},
              'conv_2': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same', 'pool': True, 'bn': True},
              'conv_3': {'num_kernels': 32, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same',  'pool': True, 'bn': True},
              'conv_4': {'num_kernels': 64, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same', 'pool': True, 'bn': True},
              }
param_fcc = {'dense_1': {'neurons': 1024, 'bn': True, 'dropout': 0.2},
             'dense_2': {'neurons': 256, 'bn': True, 'dropout': 0.2},
             #'dense_3': {'neurons': 128, 'bn': True, 'dropout': 0.2}
             }
Model = CNN.CNN(generator_training, param_conv, param_fcc,
                validation_generator=None, validation_freq=8,
                # metrics=["mae"],
                # callbacks=callbacks_list,
                use_multiprocessing=True, num_epochs=5,
                workers=12, verbose=1, model_type="regression", lr=0.0001, train=False, skip_connector=True)

history = Model.model.fit(X, y, batch_size=80, verbose=1, epochs=100, validation_data=(X_val1, y_val1),
                          shuffle=True, callbacks=callbacks_list)
np.save(path_model + "/history_100_epochs_mixed_sims.npy", history.history)
Model.model.save(path_model + "/model_100_epochs_mixed_sims.h5")


path_model = '/lfstev/deepskies/luisals/regression/train_mixed_sims/fit/skip/'

m = load_model(path_model + "/model/weights.60.hdf5")
pred1 = m.predict(X_val)
h_m_pred = output_scaler.inverse_transform(pred1).flatten()
true1 = output_scaler.inverse_transform(y_val).flatten()
np.save(path_model + "predicted1_60.npy", h_m_pred)
np.save(path_model + "true1_60.npy", true1)

pred0 = m.predict(X_val0)
h_m_pred0 = output_scaler.inverse_transform(pred0).flatten()
true0 = output_scaler.inverse_transform(y_val0).flatten()
np.save(path_model + "predicted0_60.npy", h_m_pred0)
np.save(path_model + "true0_60.npy", true0)

