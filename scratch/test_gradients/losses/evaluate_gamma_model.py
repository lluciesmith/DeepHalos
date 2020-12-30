import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
from pickle import dump, load
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf


def predict_from_model(model, epoch, gen_train, gen_val, training_IDs, training_labels_IDS, val_IDs, val_labels_IDs,
                       scaler, path_model):
    pred = model.predict_generator(gen_train, use_multiprocessing=False, workers=1, verbose=1, max_queue_size=10)
    truth_rescaled = np.array([training_labels_IDS[ID] for ID in training_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(path_model + "predicted_training_"+ epoch + ".npy", h_m_pred)
    np.save(path_model + "true_training_"+ epoch + ".npy", true)
    pred = model.predict_generator(gen_val, use_multiprocessing=False, workers=1, verbose=1, max_queue_size=10)
    truth_rescaled = np.array([val_labels_IDs[ID] for ID in val_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(path_model + "predicted_val_"+ epoch + ".npy", h_m_pred)
    np.save(path_model + "true_val_" + epoch + ".npy", true)



if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # Load data

    path_data = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/"

    scaler_output = load(open(path_data + 'scaler_output.pkl', "rb"))
    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    params_tr = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=False, **params_tr)

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)

    ######### Prepare model ##############

    kernel_reg = regularizers.l2(0.0005)
    bias_reg = regularizers.l2(0.0005)
    activation = "linear"
    relu = True

    params_all_conv = {'activation': activation, 'relu': relu, 'strides': 1, 'padding': 'same',
                       'bn': False, 'kernel_regularizer': kernel_reg, 'bias_regularizer': bias_reg}
    param_conv = {'conv_1': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': None, **params_all_conv},
                  'conv_2': {'num_kernels': 32, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_3': {'num_kernels': 64, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_4': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (3, 3, 3), 'pool': "max", **params_all_conv}
                  }

    params_all_fcc = {'bn': False, 'dropout': 0.4, 'activation': activation, 'relu': relu,
                      'kernel_regularizer': kernel_reg}
    param_fcc = {'dense_1': {'neurons': 256, **params_all_fcc},
                 'dense_2': {'neurons': 128, **params_all_fcc},
                 'last': {}
                 }

    # callbacks

    M = CNN.CNN(param_conv, param_fcc, model_type="regression", training_generator={}, metrics=['mae', 'mse'],
                num_epochs=11, dim=(31, 31, 31), max_queue_size=1, use_multiprocessing=False, workers=0, verbose=1,
                num_gpu=1, lr=0.0001, loss='mse', save_summary=False, validation_freq=1, train=False, compile=True,
                initial_epoch=10)

    m = M.model
    predictions = CNN.LossTrainableParams()(m.layers[-1].output)
    new_model = keras.Model(inputs=m.input, outputs=predictions)

    optimiser = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
    loss_c = lf.cauchy_selection_loss_fixed_boundary_trainable_gamma(new_model.layers[-1])
    new_model.compile(loss=loss_c, optimizer=optimiser)


    ############## Evaluate model #############

    path = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay/"

    epochs = ["%.2s" % i for i in np.arange(15, 65, 5)]

    # MODEL CAUCHY + SELEC + BOUNDARY

    model = path + "cauchy_selec_gamma/"
    eval = open(model + "val_callback.txt", "w")

    with tf.device("/cpu:0"):

        for epoch in epochs:

            trained_model.load_weights(model + "model/weights." + epoch + ".hdf5")
            l = trained_model.evaluate(x=generator_validation, verbose=1, workers=0, steps=50)

            eval.write(epoch + ", ")
            for item in l:
                eval.write("%s, " % item)
            eval.write(" \n")

    ############## Predict model #############

    model = path + "/cauchy_selec_bound_gamma/"

    epochs = ["15", "30", "45"]
    epochs = ["%.2s" % i for i in np.arange(15, 50, 5)]
    gamma = [0.2]

    # epochs = ["20", "35", "50"]
    with tf.device("/gpu:0"):
        for epoch in epochs:
            new_model.load_weights(model + "model/weights." + epoch + ".hdf5")
            gamma.append(new_model.layers[-1].get_weights()[0][0])
            # predict_from_model(new_model, epoch,
            #                    generator_training, generator_validation,
            #                    training_particle_IDs, training_labels_particle_IDS,
            #                    val_particle_IDs, val_labels_particle_IDS,
            #                    scaler_output, model)

    np.save(model + "gamma.npy", gamma)




