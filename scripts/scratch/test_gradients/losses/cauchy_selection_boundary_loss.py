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
                                          shuffle=True, **params_tr)

    params_val = {'batch_size': 1, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (5, 5, 5)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_particle_IDs, s.sims_dic,
                                            shuffle=False, **params_val)

    ######### TRAINING MODEL FROM MSE TRAINED ONE ##############

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay" \
                 "/cauchy_selec_bound/"

    trained_model = load_model("/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin"
                               "/larger_net/mse/model/weights.10.hdf5")

    # callbacks
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
    lrate = callbacks.LearningRateScheduler(CNN.lr_scheduler)
    callbacks_list = [checkpoint_call, csv_logger, lrate]

    lr = 0.0001
    Model = CNN.CNN({}, {}, model_type="regression", train=True, compile=True,
                    pretrained_model=trained_model, initial_epoch=10,
                    training_generator=generator_training,
                    validation_generator=generator_validation,
                    lr=0.0001, callbacks=callbacks_list, metrics=['mae', 'mse'],
                    num_epochs=100, dim=generator_validation.dim,
                    loss=lf.cauchy_selection_loss_fixed_boundary(), validation_steps=len(generator_validation),
                    max_queue_size=10, use_multiprocessing=True, workers=2, verbose=1,
                    num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1)

