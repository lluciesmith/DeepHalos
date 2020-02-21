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
import _try_numba as tg



########### CREATE GENERATORS FOR SIMULATIONS #########


# First you will have to load the simulation

# ph = "share/hypatia/lls/deep_halos/"
path_model = "/lfstev/deepskies/luisals/regression/train_mixed_sims/fit/skip/"
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
generator_1 = gbc.create_generator_sim(ids_1, mass_1, batch_size=4000, rescale_mean=rescale_mean,
                                       rescale_std=rescale_std, path=ph + "reseed1_simulation/training_set/")
X_val1, y_val1 = generator_1[0]

ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=output_scaler)
generator_0 = gbc.create_generator_sim(ids_0, mass_0, batch_size=20000, rescale_mean=rescale_mean,
                                       rescale_std=rescale_std, path=ph + "training_simulation/training_set/")
X_val0, y_val0 = generator_0[0]

dg = tg.DataGenerator()