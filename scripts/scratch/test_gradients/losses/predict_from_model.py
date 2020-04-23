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


def predict_from_model(model, epoch,
                       gen_train, gen_val, training_IDs, training_labels_IDS, val_IDs, val_labels_IDs, scaler,
                       path_model):
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

    path_transfer = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/mse/"
    scaler_training_set = load(open(path_transfer + 'scaler_output.pkl', 'rb'))

    # Create new model

    tr_set = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/cauchy_selec/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    params_inputs = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}

    training_particle_IDs = load(open(tr_set + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(tr_set + 'labels_training_set.pkl', 'rb'))
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=False, **params_inputs)

    # validation set

    validation_particle_IDs = load(open(tr_set + 'validation_set.pkl', 'rb'))
    validation_labels_particle_IDS = load(open(tr_set + 'labels_validation_set.pkl', 'rb'))
    generator_validation = tn.DataGenerator(validation_particle_IDs, validation_labels_particle_IDS, s.sims_dic,
                                            **params_inputs)

    # model_mse = load_model(path_transfer + "model/weights.10.hdf5")
    # predict_from_model(model_mse, "10",
    #                    generator_training, generator_validation,
    #                    training_particle_IDs, training_labels_particle_IDS,
    #                    validation_particle_IDs, validation_labels_particle_IDS,
    #                    scaler_training_set, path_model)

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/" \
                 "cauchy_selec_boundary/"

    for epoch in ["20", "35"]:
        model_epoch = load_model(path_model + "model/weights." + epoch + ".hdf5",
                           custom_objects={'loss':lf.cauchy_selection_loss_fixed_boundary()})
        predict_from_model(model_epoch, epoch,
                           generator_training, generator_validation,
                           training_particle_IDs, training_labels_particle_IDS,
                           validation_particle_IDs, validation_labels_particle_IDS,
                           scaler_training_set, path_model)
        del model_epoch
        gc.collect()


    # Not sure this runs...

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net" \
                 "/cauchy_selec/train_gamma/"

    for epoch in ["20", "35"]:
        model_epoch = load_model(path_model + "model/weights." + epoch + ".hdf5",
                                 custom_objects={'loss':lf.cauchy_selection_loss_trainable_gamma(CNN.CauchyLayer()),
                                                 'CauchyLayer': CNN.CauchyLayer})
        predict_from_model(model_epoch, epoch,
                           generator_training, generator_validation,
                           training_particle_IDs, training_labels_particle_IDS,
                           validation_particle_IDs, validation_labels_particle_IDS,
                           scaler_training_set, path_model)
        del model_epoch
        gc.collect()

