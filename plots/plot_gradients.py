import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as k
from dlhalos_code import loss_functions as lf
from dlhalos_code import CNN
import sys
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np

saving_path = "/mnt/beegfs/work/ati/pearl037/regression/full_mass_range/200k_random_training/9sims/Xavier/"
num_epoch = "07"

path_sims = "/mnt/beegfs/work/ati/pearl037/"
all_sims = ["0", "1", "2", "4", "5", "7", "8", "9", "10", "6"]
s = tn.SimulationPreparation(all_sims, path=path_sims)

path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set/9sims/random/200k/"
training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

# Create the generators for training

dim = (75, 75, 75)
params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                      shuffle=True, **params_tr)
params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                        shuffle=False, **params_val)

######### TRAIN THE MODEL ################

alpha = 10**(-3.5)
params_all_conv = {'activation': "tanh", 'relu': False, 'strides': 1, 'padding': 'same', 'bn': False,
                   'kernel_regularizer': reg.l2_norm(alpha)
                   }
param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
              'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
              }

# Dense layers parameters

params_all_fcc = {'bn': False, 'activation': "tanh", 'relu': False,
                  'kernel_regularizer': reg.l1_and_l21_group(alpha)}
param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
             'last': {}}

# Train for one epoch using MSE loss and the rest using a Cauchy loss

# weights = saving_path + "model/weights." + num_epoch + ".h5"
Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator=generator_training,
                      shuffle=True, validation_generator=generator_validation, num_epochs=100,
                      dim=generator_validation.dim, initialiser="Xavier_uniform", max_queue_size=10,
                      use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=0.0001, save_summary=False,
                      path_summary=saving_path, validation_freq=1, train=False, compile=True, initial_epoch=None)


w = Model.model.trainable_weights
out = Model.model.output
# loss_c = lf.cauchy_selection_loss_fixed_boundary_trainable_gamma(Model.model.layers[-1])
loss_c = lf.cauchy_selection_loss_fixed_boundary()
g = generator_validation[0]
labels = np.float32(g[1].reshape(len(g[1]),1))

l = loss_c(labels, out)
gradients = k.gradients(l, w)
sess = k.get_session()
evaluated_gradients = sess.run(gradients, feed_dict={Model.model.input: np.float32(g[0])})
np.save(saving_path + "gradients_epoch_" + num_epoch + ".npy", evaluated_gradients)
np.save(saving_path + "weights_epoch_" + num_epoch + ".npy", Model.model.get_weights())


tr_weights = Model.model.trainable_weights
tr_g = Model.model.optimizer.get_gradients(Model.model.total_loss, tr_weights)
get_g = k.function(inputs=[Model.model._feed_inputs, Model.model._feed_targets], outputs=tr_g)
output_grad = get_g([np.float32(g[0]), labels])
np.save(saving_path + "adam_gradients_epoch_" + num_epoch + ".npy", output_grad)

all_weights = Model.model.get_weights()
trainable_weights = [index for index,w in enumerate(Model.model.weights) if w in Model.model.trainable_weights]
w1_eval = [all_weights[i] for i in trainable_weights]
np.save(saving_path + "weights_epoch_" + num_epoch + ".npy", w1_eval)