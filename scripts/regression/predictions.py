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
from pickle import load


########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########


# First choose the correct path to the model and the parameters you used
# during training

path_model = "/lfstev/deepskies/luisals/regression/ics_res121/"

num_epoch = "25"
validation_sims = ["1"]
batch_size = 80
rescale_mean = 1.005
rescale_std = 0.05050
dim = (75, 75, 75)

# load validation set

s = tn.SimulationPreparation(validation_sims)
scaler_output = load(open(path_model + 'scaler_output.pkl', 'rb'))

validation_set = tn.InputsPreparation(validation_sims, load_ids=True, scaler_output=scaler_output, shuffle=False)
generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
                                        batch_size=batch_size, rescale_mean=rescale_mean, rescale_std=rescale_std,
                                        dim=dim)

# load model

# model = load_model(path_model + "/model_100_epochs_mixed_sims.h5")
model = load_model(path_model + "/model/weights." + num_epoch + " .hdf5")

pred = model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
truth_rescaled = np.array([val for (key, val) in validation_set.labels_particle_IDS.items()])

h_m_pred = scaler_output.inverse_transform(pred).flatten()
true1 = scaler_output.inverse_transform(truth_rescaled).flatten()

np.save(path_model + "/predicted1_" + num_epoch + ".npy", h_m_pred)
np.save(path_model + "/true1_" + num_epoch + ".npy", true1)

