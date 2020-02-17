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
path_model = "/lfstev/deepskies/luisals/regression/ics_res75"
ph = "/lfstev/deepskies/luisals/"
dim = (75, 75, 75)

t0 = time.time()

######### COLLECT THE DATA ##############

# training set

f = "random_training_set.txt"
ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=None,shuffle=False)
ids_2, mass_2 = gbc.get_ids_and_regression_labels(sim="2", ids_filename=f, fitted_scaler=None, shuffle=False)
ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename=f, fitted_scaler=None, shuffle=False)
ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename=f, fitted_scaler=None, shuffle=False)
ids_6, mass_6 = gbc.get_ids_and_regression_labels(sim="6", ids_filename=f, fitted_scaler=None, shuffle=False)

sims = ["0", "2", "4", "5", "6"]
ids_s = [ids_0, ids_2, ids_4, ids_5, ids_6]
mass_ids = [mass_0, mass_2, mass_4, mass_5, mass_6]
output_ids, output_scaler = gbc.get_standard_scaler_and_transform([mass_0, mass_2, mass_4, mass_5, mass_6])
generator_training = gbc.create_generator_multiple_sims(sims, ids_s, output_ids, batch_size=100,
                                                        rescale_mean=0, rescale_std=1, dim=dim, z=99)
X, y = generator_training[0]

rescale_mean = X[:1000].mean()
rescale_std = X[:1000].std()
print("The mean value of the training set is " + str(rescale_mean))
print("The std value of the training set is " + str(rescale_std))
X_rescaled = (X - rescale_mean)/rescale_std
#
# def load():
#     generator_training = gbc.create_generator_multiple_sims(sims, ids_s, output_ids, batch_size=60000,
#                                                             rescale_mean=0, rescale_std=1, dim=dim, z=99)
#     X, y = generator_training[0]
#     return X, y
#
# generator_training = gbc.create_generator_multiple_sims(sims, ids_s, output_ids, batch_size=100000,
#                                                         rescale_mean=0, rescale_std=1, dim=dim, z=99)
# X, y = generator_training[0]
# np.save("/lfstev/deepskies/luisals/20000_features.npy", X)
#
# def load_2():
#     return np.load("/lfstev/deepskies/luisals/20000_features.npy")
#
# timeit.timeit(load, number=10)/10
# timeit.timeit(load_2, number=10)/10



# validation set

ran_val = np.random.choice(np.arange(20000), 4000)
np.save(path_model + "ran_val1.npy", ran_val)

ids_1, mass_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename=f, fitted_scaler=output_scaler)
generator_1 = gbc.create_generator_sim(ids_1, mass_1, batch_size=4000, dim=dim, z=99,
                                       rescale_mean=rescale_mean, rescale_std=rescale_std,
                                       path=ph + "reseed1_simulation/training_set_res75/")
X_val1, y_val1 = generator_1[0]
X_val1_rescaled = (X_val1 - rescale_mean)/rescale_std

t1 = time.time()
print("Loading generators took " + str((t1 - t0) / 60) + " minutes.")

######### TRAINING MODEL ##############

# checkpoint
filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

# save histories
csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=False)

# decay the learning rate
lr_decay = LearningRateScheduler(CNN.lr_scheduler)

callbacks_list = [checkpoint_call, csv_logger]
# callbacks_list = [checkpoint_call, csv_logger, lr_decay]

tensorflow.compat.v1.set_random_seed(7)

param_conv = {'conv_1': {'num_kernels': 8, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same', 'pool': True, 'bn': True},
              'conv_2': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same', 'pool': True, 'bn': True},
              'conv_3': {'num_kernels': 32, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same',  'pool': True, 'bn': True},
              'conv_4': {'num_kernels': 64, 'dim_kernel': (3, 3, 3),
                         'strides': 1, 'padding': 'same', 'pool': True, 'bn': True},
              'conv_5': {'num_kernels': 16, 'dim_kernel': (1, 1, 1),
                         'strides': 1, 'padding': 'same', 'pool': False, 'bn': True},
              }

param_fcc = {'dense_1': {'neurons': 1024, 'bn': True, 'dropout': 0.4},
             'dense_2': {'neurons': 256, 'bn': True, 'dropout': 0.4},
             }

Model = CNN.CNN(generator_training, param_conv, param_fcc,
                validation_generator=None, validation_freq=8, use_multiprocessing=True, num_epochs=5,
                workers=12, verbose=1, model_type="regression", lr=0.0001, train=False, skip_connector=False)

history = Model.model.fit(X_rescaled, y, batch_size=80, verbose=1, epochs=100, validation_data=(X_val1_rescaled, y_val1),
                          shuffle=True, callbacks=callbacks_list, initial_epoch=32)

np.save(path_model + "/history_100_epochs_mixed_sims.npy", history.history)
Model.model.save(path_model + "/model_100_epochs_mixed_sims.h5")

######### TRAINING MODEL ##############

path_model = "/lfstev/deepskies/luisals/regression/ics_res75"

ids_1, mass_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename=f, fitted_scaler=output_scaler, shuffle=False)
generator_1 = gbc.create_generator_sim(ids_1, mass_1, batch_size=20000, dim=dim, z=99,
                                       rescale_mean=rescale_mean, rescale_std=rescale_std,
                                       path=ph + "reseed1_simulation/training_set_res75/")
X_val1, y_val1 = generator_1[0]
X_val1_rescaled = (X_val1 - rescale_mean)/rescale_std

m = load_model(path_model + "/model/weights.60.hdf5")
pred1 = m.predict(X_val1_rescaled)
h_m_pred = output_scaler.inverse_transform(pred1).flatten()
true1 = output_scaler.inverse_transform(y_val1).flatten()
np.save(path_model + "/predicted1_60.npy", h_m_pred)
np.save(path_model + "/true1_60.npy", true1)

