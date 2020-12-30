import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import custom_regularizers as reg
import dlhalos_code.data_processing as tn
from pickle import dump, load
import matplotlib.pyplot as plt


########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

# Load data

path_data = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/"

scaler_output = load(open(path_data + 'scaler_output_50000.pkl', "rb"))
training_particle_IDs = load(open(path_data + 'training_set_50000.pkl', 'rb'))
training_labels_particle_IDS = load(open(path_data + 'labels_training_set_50000.pkl', 'rb'))
val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

# Create the generators for training

all_sims = ["0", "1", "2", "4", "5", "6"]
s = tn.SimulationPreparation(all_sims)

params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                      shuffle=True, **params_tr)

params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                        shuffle=True, **params_val)

######### TRAIN THE MODEL ################

path = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
       "/cauchy_selec_bound_gamma_train_alpha/l2_conv_l21_l1_dense/test/"

log_alpha = -4
alpha = 0.0001
path_model = path + "log_alpha_" + str(log_alpha) + "/no_reg_mse_epoch/"
path_model1 = path_model + "reg_added_layer/"
path_model2 = path_model + "reg_added_loss/"

############### TEST 1 ########################

# Convolutional layers

params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                   'kernel_regularizer': reg.l2_norm(alpha)}
param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
              'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
              'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
              }

# Dense layers parameters

params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True, 'kernel_regularizer': reg.l1_and_l21_group(alpha)}
param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
             'last': {}}
# Regularization parameters + Cauchy likelihood

reg_params = {'init_gamma': 0.2}

# Train for 100 epochs

Model1 = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator={}, validation_generator={},
                       num_epochs=20, dim=(31, 31, 31), max_queue_size=10, use_multiprocessing=False, workers=0,
                       verbose=1, num_gpu=1, lr=0.0001, save_summary=True, path_summary="./reg_added_layer/",
                       validation_freq=1, train=False, compile=True)


Model2 = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator={}, validation_generator={},
                       num_epochs=20, dim=(31, 31, 31), max_queue_size=10, use_multiprocessing=False, workers=0,
                       verbose=1, num_gpu=1, lr=0.0001, save_summary=True, path_summary="./reg_added_loss/",
                       validation_freq=1, train=False, compile=True)


# Compare the weights of every layer in the two models


def get_weights(model):
    weights = []
    names_layers = [layer.name for layer in model.layers]
    conv_layers = [s for s in names_layers if 'conv3d' in s]
    dense_layers = [s for s in names_layers if 'dense' in s]
    all_layers = conv_layers + dense_layers
    indices = [i for i, item in enumerate(names_layers) if item in all_layers]
    for index in indices:
        l = model.layers[index]
        w, b = l.get_weights()
        weights.append(w)
    return weights, all_layers

w1, layers_names1 = get_weights(Model1.model)
w2, layers_names2 = get_weights(Model2.model)

for i in range(7):
    f = plt.figure()
    _ = plt.hist(w1[i].flatten(), histtype="step", lw=1.5, label=layers_names1[i])
    __ = plt.hist(w2[i].flatten(), histtype="step", lw=1.5)
    plt.legend(loc="best")
    plt.xlabel("w")


# l1 = Model1.model.evaluate(generator_training, verbose=1)
# l2 = Model2.model.evaluate(generator_training, verbose=1)
#
# y = tf.convert_to_tensor(y_true.reshape(len(y_true), 1), dtype="float32")
# yp = tf.convert_to_tensor(y_pred, dtype="float32")
#
# l2_cauchy = lf.cauchy_selection_loss_fixed_boundary()(y, yp)
# l1_cauchy = lf.cauchy_selection_loss_fixed_boundary()(y_true, y_pred)
#
# lr_reg = [(K.get_value(Model2.model.losses[i])) for i in range(7)]
#
# Model1.model.layers[indices[0]].losses
# Model2.model.layers[indices[0]].losses