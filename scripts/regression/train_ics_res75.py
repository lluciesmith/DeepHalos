import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from utils import data_loading as dl
import time
import tensorflow

if __name__ == "__main__":

    ########### CREATE GENERATORS FOR SIMULATIONS #########

    # ph = "share/hypatia/lls/deep_halos/"
    print("start script")
    path_model = "/lfstev/deepskies/luisals/regression/ics_res75/"
    ph = "/lfstev/deepskies/luisals/"
    dim = (75, 75, 75)

    t0 = time.time()

    ######### COLLECT THE DATA ##############

    # training set

    train_sims = ["0", "2", "4", "5", "6"]
    val_sims = ["1"]
    print("start data processing")
    dp = dl.DataProcessing(train_sims, val_sims)

    ind = np.random.choice(range(len(dp.y_test)), 4000, replace=False)
    X_val = dp.X_test[ind]
    y_val = dp.y_test[ind]


    ######### TRAINING MODEL ##############

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

    # save histories
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=False)

    # decay the learning rate
    lr_decay = LearningRateScheduler(CNN.lr_scheduler)

    # callbacks_list = [checkpoint_call, csv_logger]
    callbacks_list = [checkpoint_call, csv_logger, lr_decay]

    tensorflow.compat.v1.set_random_seed(7)

    param_conv = {'conv_1': {'num_kernels': 4, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  'conv_2': {'num_kernels': 8, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  'conv_3': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same',  'pool': "max", 'bn': True},
                  'conv_4': {'num_kernels': 4, 'dim_kernel': (1, 1, 1),
                              'strides': 1, 'padding': 'same', 'pool': None, 'bn': True}
                  }

    param_fcc = {#'dense_1': {'neurons': 1024, 'bn': False, 'dropout': 0.2},
                 'dense_1': {'neurons': 256, 'bn': True, 'dropout': 0.4},
                 'dense_2': {'neurons': 128, 'bn': False, 'dropout': 0.4}}

    Model = CNN.CNN(param_conv, param_fcc, dim=(75, 75, 75),
                    training_generator=None, validation_generator=None, validation_freq=1,
                    callbacks=callbacks_list, use_multiprocessing=True, num_epochs=80,
                    workers=12, verbose=1, model_type="regression", lr=0.0001, train=False)

    history = Model.model.fit(dp.X_train, dp.y_train, batch_size=80, verbose=1, epochs=100,
                              validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks_list)

    np.save(path_model + "/history_100_epochs_mixed_sims.npy", history.history)
    Model.model.save(path_model + "/model_100_epochs_mixed_sims.h5")

    ######### TESTING ############

    pred1 = Model.model.predict(dp.X_test)
    h_m_pred = dp.output_scaler.inverse_transform(pred1).flatten()
    true1 = dp.output_scaler.inverse_transform(dp.y_test).flatten()
    np.save(path_model + "/predicted1_100.npy", h_m_pred)
    np.save(path_model + "/true1_100.npy", true1)

