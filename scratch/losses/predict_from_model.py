import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import dlhalos_code.data_processing as tn
from pickle import dump, load
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import load_model
import gc
import tensorflow as tf


def predict_from_model(model, epoch, gen_train, gen_val, training_IDs, training_labels_IDS,
                       val_IDs, val_labels_IDs, scaler, path_model):
    pred = model.predict_generator(gen_train, use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10)
    truth_rescaled = np.array([training_labels_IDS[ID] for ID in training_IDs])
    h_m_pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    true = scaler.inverse_transform(truth_rescaled.reshape(-1, 1)).flatten()
    np.save(path_model + "predicted_training_"+ epoch + ".npy", h_m_pred)
    np.save(path_model + "true_training_"+ epoch + ".npy", true)
    pred = model.predict_generator(gen_val, use_multiprocessing=False, workers=0, verbose=1, max_queue_size=10)
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

    ################ LOAD MODEL #######################

    path = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay/"

    epochs = ["15", "30", "45"]

    # MODEL CAUCHY + SELEC + BOUNDARY

    path_model = path + "cauchy_selec_bound_test/"

    with tf.device("/gpu:0"):
        for epoch in epochs:
                model_epoch = load_model(path_model + "model/weights." + epoch + ".hdf5",
                                   custom_objects={'loss':lf.cauchy_selection_loss_fixed_boundary()})
                predict_from_model(model_epoch, epoch,
                                   generator_training, generator_validation,
                                   training_particle_IDs, training_labels_particle_IDS,
                                   val_particle_IDs, val_labels_particle_IDS,
                                   scaler_output, path_model)
                del model_epoch
                gc.collect()

    # MODEL CAUCHY + SELEC

    # path_model = path + "cauchy_selec/"
    path_model = path + "cauchy_selec_relu_last_act/"

    with tf.device("/gpu:0"):
        for epoch in epochs:
            model_epoch = load_model(path_model + "model/weights." + epoch + ".hdf5",
                                     custom_objects={'loss':lf.cauchy_selection_loss()})
            predict_from_model(model_epoch, epoch,
                               generator_training, generator_validation,
                               training_particle_IDs, training_labels_particle_IDS,
                               val_particle_IDs, val_labels_particle_IDS,
                               scaler_output, path_model)
            del model_epoch
            gc.collect()




