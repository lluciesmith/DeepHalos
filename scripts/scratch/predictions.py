import sys
sys.path.append("/home/luisals/DeepHalos")
from tensorflow.keras.models import load_model
import dlhalos_code.data_processing as tn
import numpy as np


############### PREDICT ############

all_sims = ["0", "1", "3", "4", "5"]
s = tn.SimulationPreparation(all_sims)

training_sims = ["0", "3", "4", "5"]
validation_sims = ["1"]
# batch_size = 80
rescale_mean = 1.005
rescale_std = 0.05050

training_set = tn.InputsPreparation(training_sims, load_ids=True)
generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
                                      batch_size=80000, rescale_mean=rescale_mean, rescale_std=rescale_std)

output_scaler = training_set.labels_scaler

validation_set = tn.InputsPreparation(validation_sims, load_ids=True, scaler_output=output_scaler)
generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS, s.sims_dic,
                                        batch_size=20000, rescale_mean=rescale_mean, rescale_std=rescale_std)
X_val1, y_val1 = generator_validation[0]

path_model = '/lfstev/deepskies/luisals/scratch2/'

m = load_model(path_model + "/model/weights.100.hdf5")
pred1 = m.predict(X_val1)
h_m_pred = output_scaler.inverse_transform(pred1).flatten()
true1 = output_scaler.inverse_transform(y_val1).flatten()
np.save(path_model + "predicted1_100.npy", h_m_pred)
np.save(path_model + "true1_100.npy", true1)