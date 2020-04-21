import sys

sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
from dlhalos_code import loss_functions as lf
import tensorflow.keras as keras
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

    path_transfer = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/mse/"
    scaler_training_set = load(open(path_transfer + 'scaler_output.pkl', 'rb'))

    # Create new model

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/" \
                 "cauchy_selec_boundary/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    # Use the same training set/validation set as for cauchy+selection loss function

    params_inputs = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    tr_set = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/cauchy_selec/"

    training_particle_IDs = load(open(tr_set + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(tr_set + 'labels_training_set.pkl', 'rb'))
    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=False, **params_inputs)

    # # validation set
    #
    # validation_particle_IDs = load(open(tr_set + 'validation_set.pkl', 'rb'))
    # validation_labels_particle_IDS = load(open(tr_set + 'labels_validation_set.pkl', 'rb'))
    # generator_validation = tn.DataGenerator(validation_particle_IDs, validation_labels_particle_IDS, s.sims_dic,
    #                                         **params_inputs)

    ######### TRAINING MODEL FROM MSE TRAINED ONE ##############

    trained_model = load_model(path_transfer + "model/weights.10.hdf5")

    # callbacks
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)
    callbacks_list = [checkpoint_call, csv_logger]

    lr = 0.0001
    Model = CNN.CNN({}, {}, model_type="regression", train=True, compile=True,
                    pretrained_model=trained_model, initial_epoch=10,
                    training_generator=generator_training,
                    # validation_generator=generator_validation,
                    lr=lr, callbacks=callbacks_list, metrics=['mae', 'mse'],
                    num_epochs=100, dim=params_inputs['dim'],
                    loss=lf.cauchy_selection_loss_fixed_boundary(), validation_steps=50,
                    max_queue_size=10, use_multiprocessing=False, workers=1, verbose=1,
                    num_gpu=1, save_summary=True,  path_summary=path_model, validation_freq=1)

