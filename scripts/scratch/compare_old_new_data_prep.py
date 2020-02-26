import sys
sys.path.append("/home/luisals/DeepHalos")
import dlhalos_code.data_processing as tn
from utils.old import generators_training as gbc
import time
import numpy as np

all_sims = ["0", "1", "3", "4", "5"]
s = tn.SimulationPreparation(all_sims)

training_sims = ["0", "3", "4", "5"]
training_set_new = tn.InputsPreparation(training_sims, load_ids=True, ids_filename="random_training_set.txt",
                                        random_subset_each_sim=None, path="/lfstev/deepskies/luisals/",
                                        scaler_output=None, return_rescaled_outputs=True, shuffle=True)


rescale_mean = 1.004
rescale_std = 0.05

t0 = time.time()

f = "random_training_set.txt"
ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=None, shuffle=False)
ids_3, mass_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename=f, fitted_scaler=None, shuffle=False)
ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename=f, fitted_scaler=None, shuffle=False)
ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename=f, fitted_scaler=None, shuffle=False)

# training set
sims = ["0", "3", "4", "5"]
ids_s = [ids_0, ids_3, ids_4, ids_5]
mass_ids = [mass_0, mass_3, mass_4, mass_5]
output_ids, output_scaler = gbc.get_standard_scaler_and_transform([mass_0, mass_3, mass_4, mass_5])
generator_training = gbc.create_generator_multiple_sims(sims, ids_s, output_ids, batch_size=80000,
                                                        rescale_mean=0, rescale_std=1)
# tests

np.allclose(output_scaler.scale_, training_set_new.scaler_output.scale_)
np.allclose(output_scaler.var_, training_set_new.scaler_output.var_)
np.allclose(output_scaler.mean_, training_set_new.scaler_output.mean_)

print(training_set_new.particle_IDs == generator_training.list_IDs)
print(generator_training.labels['sim-5-id-388367'] == training_set_new.labels_particle_IDS['sim-5-id-388367'])
