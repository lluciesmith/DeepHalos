import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from utils.old import generators_training as gbc
import time
import tensorflow
from keras.model import load_model


if __name__ == "__main__":
    ########### CREATE GENERATORS FOR SIMULATIONS #########

    # ph = "share/hypatia/lls/deep_halos/"
    # path_model = "/lfstev/deepskies/luisals/regression/z0/alpha0.03_5sims/batchnorm"
    path_model = "/lfstev/deepskies/luisals/regression/z0/z0_res75"
    ph = "/lfstev/deepskies/luisals/"

    # rescale_mean = 1.239
    # rescale_std = 0.89
    rescale_mean = 0
    rescale_std = 1

    t0 = time.time()

    f = "random_training_set.txt"
    ids_0, mass_0 = gbc.get_ids_and_regression_labels(sim="0", ids_filename=f, fitted_scaler=None)
    ids_2, mass_2 = gbc.get_ids_and_regression_labels(sim="2", ids_filename=f, fitted_scaler=None)
    ids_3, mass_3 = gbc.get_ids_and_regression_labels(sim="3", ids_filename=f, fitted_scaler=None)
    ids_4, mass_4 = gbc.get_ids_and_regression_labels(sim="4", ids_filename=f, fitted_scaler=None)
    ids_5, mass_5 = gbc.get_ids_and_regression_labels(sim="5", ids_filename=f, fitted_scaler=None)

    # training set
    sims = ["0", "2", "3", "4", "5"]
    ids_s = [ids_0, ids_2, ids_3, ids_4, ids_5]
    output_ids, output_scaler = gbc.get_standard_scaler_and_transform([mass_0, mass_2, mass_3, mass_4, mass_5])

    generator_training = gbc.create_generator_multiple_sims(sims, ids_s, output_ids, batch_size=80, dim=(75, 75, 75),
                                                            rescale_mean=rescale_mean, rescale_std=rescale_std, z=0)

    # validation set
    ran_val = np.random.choice(np.arange(20000), 4000)
    # np.save(path_model + "ran_val1.npy", ran_val)
    ids_1, mass_1 = gbc.get_ids_and_regression_labels(sim="1", ids_filename=f, fitted_scaler=output_scaler)
    generator_1 = gbc.create_generator_sim(ids_1[ran_val], mass_1[ran_val], batch_size=80, dim=(75, 75, 75),
                                           rescale_mean=rescale_mean, rescale_std=rescale_std, z=0,
                                           path=ph + "reseed1_simulation/z0_subboxes_res75/")

    t1 = time.time()
    print("Loading generators took " + str((t1 - t0) / 60) + " minutes.")

######### TRAINING MODEL ##############

    model = load_model(path_model + "/model/weights.25.hdf5")

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, freq='epoch')

    # save histories
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)

    # decay the learning rate
    lr_decay = LearningRateScheduler(CNN.lr_scheduler)

    # callbacks_list = [checkpoint_call, csv_logger]
    callbacks_list = [checkpoint_call, csv_logger, lr_decay]

    tensorflow.compat.v1.set_random_seed(7)

    history = model.fit_generator(generator=generator_training, validation_data=generator_1,
                                  use_multiprocessing=True, workers=12,
                                  verbose=1, epochs=50, shuffle=True,
                                  callbacks=callbacks_list, validation_freq=1, initial_epoch=25)

    np.save(path_model + "/history_80_epochs_mixed_sims.npy", history)
    model.save(path_model + "/model_80_epochs_mixed_sims.h5")

