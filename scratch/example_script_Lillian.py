import numpy as np
import matplotlib.pyplot as plt

from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn

############### PREPARE THE DATA ###############

# Specify the paths where you keep the data

path_sims = "/Users/lls/Documents/Lillian/"
saving_path = "/Users/lls/Documents/Lillian/example/"

# Prepare the simulations you will use for training/validation

sim_id = ["0"]
s = tn.SimulationPreparation(sim_id, path=path_sims)

# Prepare the training data -- select particles, get their labels -- and generate a ``generator'' which will be used as
# input to the CNN

tr_set = tn.InputsPreparation(sim_id, shuffle=True, scaler_type="minmax", return_rescaled_outputs=True, load_ids=False,
                              output_range=(-1, 1), random_style="random", random_subset_all=1000, path=path_sims)
params_tr = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (11, 11, 11)}
generator_training = tn.DataGenerator(tr_set.particle_IDs, tr_set.labels_particle_IDS, s.sims_dic, shuffle=False,
                                      **params_tr)

# Do the same for the validation set

v_set = tn.InputsPreparation(sim_id, scaler_type="minmax", shuffle=True, random_style="random", load_ids=False,
                             random_subset_all=1000, scaler_output=tr_set.scaler_output, path=path_sims)
params_val = {'batch_size': 10, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (11, 11, 11)}
generator_validation = tn.DataGenerator(v_set.particle_IDs, v_set.labels_particle_IDS, s.sims_dic, shuffle=False,
                                        **params_val)


######### TRAIN THE MODEL ################

# Define the parameters of the convolutional layers

params_all_conv = {'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                   'kernel_regularizer': reg.l2_norm(10**-4)}
param_conv = {'conv_1': {'num_kernels': 5, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_2': {'num_kernels': 5, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              }

# Define the parameters of the fully-connected layers

params_all_fcc = {'bn': False, 'relu': True, 'kernel_regularizer': reg.l1_and_l21_group(10**-4)}
param_fcc = {'dense_1': {'neurons': 100, **params_all_fcc},
             'dense_2': {'neurons': 20, **params_all_fcc},
             'last': {}}

# Train the model

Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression",
                      training_generator=generator_training, validation_generator=generator_validation,
                      steps_per_epoch=len(generator_training), validation_steps=len(generator_validation),
                      num_epochs=30, dim=generator_training.dim, initialiser="Xavier_uniform", max_queue_size=10,
                      use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=10**-3, save_summary=True,
                      path_summary=saving_path, validation_freq=1, train=True, compile=True)


######### EVALUATE THE MODEL ################

# Checkout what your model looks like

print(Model.model.summary())

# Comparing the predictions with the ground truth labels for the training set and the test set

# The training set predictions should look good!

pred_rescaled = Model.model.predict_generator(generator_training, use_multiprocessing=False, workers=1, verbose=1)
truth_rescaled = np.array([val for (key, val) in tr_set.labels_particle_IDS.items()])
predictions = tr_set.scaler_output.inverse_transform(pred_rescaled.reshape(-1, 1)).flatten()
ground_truth = tr_set.scaler_output.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()

plt.scatter(ground_truth, predictions, s=1, label="training set")
plt.xlabel("Ground truth")
plt.ylabel("Predictions")
plt.legend(loc="best")

# Your validation set prediction won't look as good... but that's only because the training set is suboptimal and
# there are not enough layers in the network.

pred_rescaled_val = Model.model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
truth_rescaled_val = np.array([val for (key, val) in v_set.labels_particle_IDS.items()])
predictions_val = tr_set.scaler_output.inverse_transform(pred_rescaled_val.reshape(-1, 1)).flatten()
ground_truth_val = tr_set.scaler_output.inverse_transform(truth_rescaled_val.reshape(-1, 1)).flatten()

plt.scatter(ground_truth_val, predictions_val, s=1, label="validation set")
plt.legend(loc="best")