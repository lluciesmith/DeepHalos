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

    all_sims = ["6"]
    s = tn.SimulationPreparation(all_sims)

    params_val = {'batch_size': 100, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': (31, 31, 31)}
    generator_validation = tn.DataGenerator(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params_val)


    ################ LOAD MODEL #######################

    path = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/lr_decay/"

    epochs = ["%.2s" % i for i in np.arange(15, 75, 5)]

    # MODEL CAUCHY + SELEC + BOUNDARY
    #
    # model = path + "cauchy_selec_bound/"
    # eval = open(model + "val_callback.txt", "w")
    # eval.write("Epoch, val_loss, val_mae, val_mse \n")
    #
    # with tf.device("/cpu:0"):
    #     for epoch in epochs:
    #             model_epoch = load_model(model + "model/weights." + epoch + ".hdf5",
    #                                custom_objects={'loss':lf.cauchy_selection_loss_fixed_boundary()})
    #
    #             l = model_epoch.evaluate(x=generator_validation, verbose=1, workers=0, steps=50)
    # 
    #             eval.write(epoch + ", ")
    #             for item in l:
    #                 eval.write("%s, " % item)
    #             eval.write(" \n")
    #
    #             del model_epoch
    #             gc.collect()

    # MODEL CAUCHY + SELEC

    model = path + "cauchy_selec/"
    eval = open(model + "val_callback.txt", "w")
    eval.write("Epoch, val_loss, val_mae, val_mse \n")

    with tf.device("/cpu:0"):
        for epoch in epochs:
                model_epoch = load_model(model + "model/weights." + epoch + ".hdf5",
                                   custom_objects={'loss':lf.cauchy_selection_loss()})

                l = model_epoch.evaluate(x=generator_validation, verbose=1, workers=0, steps=50)

                eval.write(epoch + ", ")
                for item in l:
                    eval.write("%s, " % item)
                eval.write(" \n")

                del model_epoch
                gc.collect()

