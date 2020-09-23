from dlhalos_code import CNN
import dlhalos_code.data_processing as tn
from pickle import load
import numpy as np
from dlhalos_code import custom_regularizers as reg


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    saving_path = "/mnt/beegfs/work/ati/pearl037/regression/mass_range_13.4/random_20sims_200K/lr_5e-5/alpha_-2.2" \
                  "/training_set2/"

    s = tn.SimulationPreparation(["6"], path="/mnt/beegfs/work/ati/pearl037/")
    path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set_13.4/20sims/random/200k/"
    val_particle_IDs = load(open(path_data + 'larger_validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    dim = (75, 75, 75)
    params_val = {'batch_size': 50, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                        shuffle=False, **params_val)

    ######### TRAIN THE MODEL ################

    alpha = 10**-2.2
    params_all_conv = {'activation': "linear", 'relu': True, 'strides': 1, 'padding': 'same', 'bn': False,
                       'kernel_regularizer': reg.l2_norm(alpha)
                       }
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    params_all_fcc = {'bn': False, 'activation': "linear", 'relu': True,
                      'kernel_regularizer': reg.l1_and_l21_group(alpha)}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc}, 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}}

    lr = 0.00005
    Model = CNN.CNNCauchy(param_conv, param_fcc, model_type="regression", training_generator={}, shuffle=True,
                          validation_generator=generator_validation, num_epochs=100, dim=generator_validation.dim,
                          max_queue_size=80, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=lr,
                          save_summary=False, path_summary=saving_path, validation_freq=1, train=False, compile=True,
                          initial_epoch=None, initialiser="Xavier_uniform")

    epochs = ["%02d" % num for num in np.arange(2, 19, step=2)]
    loss_val = []

    for num_epoch in epochs[:-3]:
        weights = saving_path + "model/weights." + num_epoch + ".h5"
        Model.model.load_weights(weights)

        l_v = Model.model.evaluate_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1,
                                             steps=len(generator_validation))
        loss_val.append(l_v)

    np.save(saving_path + "loss_larger_validation_set.npy", loss_val)
