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
    np.save(path_model + "true_val_"+ epoch + ".npy", true)

if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    path_transfer = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/mse/"
    scaler_training_set = load(open(path_transfer + 'scaler_output.pkl', 'rb'))

    # Create new model

    path_model = "/lfstev/deepskies/luisals/regression/large_CNN/test_lowmass/reg_10000_perbin/larger_net/cauchy_selec/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    params_inputs = {'batch_size': 100,
                     'rescale_mean': 1.005,
                     'rescale_std': 0.05050,
                     'dim': (31, 31, 31),
                     # 'shuffle': True
                     }

    try:
        training_particle_IDs = load(open(path_model + 'training_set.pkl', 'rb'))
        training_labels_particle_IDS = load(open(path_model + 'labels_training_set.pkl', 'rb'))

        print("loaded training set")
    except OSError:
        training_set = tn.InputsPreparation(train_sims, scaler_type="minmax", output_range=(-1, 1),
                                            load_ids=False, shuffle=True,
                                            log_high_mass_limit=13,
                                            random_style="uniform", random_subset_each_sim=1000000,
                                            num_per_mass_bin=1000, scaler_output=scaler_training_set)
        dump(training_set.particle_IDs, open(path_model + 'training_set.pkl', 'wb'))
        dump(training_set.labels_particle_IDS, open(path_model + 'labels_training_set.pkl', 'wb'))

        training_particle_IDs = training_set.particle_IDs
        training_labels_particle_IDS = training_set.labels_particle_IDS
        s_output = training_set.scaler_output

    generator_training = tn.DataGenerator(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                          shuffle=False, **params_inputs)

    try:
        validation_particle_IDs = load(open(path_model + 'validation_set.pkl', 'rb'))
        validation_labels_particle_IDS = load(open(path_model + 'labels_validation_set.pkl', 'rb'))
        print("loaded validation set")
    except OSError:
        validation_set = tn.InputsPreparation([val_sim], load_ids=False, random_subset_each_sim=20000,
                                              log_high_mass_limit=13, scaler_output=scaler_training_set)
        dump(validation_set.particle_IDs, open(path_model + 'validation_set.pkl', 'wb'))
        dump(validation_set.labels_particle_IDS, open(path_model + 'labels_validation_set.pkl', 'wb'))
        validation_particle_IDs = validation_set.particle_IDs
        validation_labels_particle_IDS = validation_set.labels_particle_IDS

    generator_validation = tn.DataGenerator(validation_particle_IDs, validation_labels_particle_IDS, s.sims_dic,
                                            **params_inputs)

    # model_mse = load_model(path_transfer + "model/weights.10.hdf5")
    # predict_from_model(model_mse, "10",
    #                    generator_training, generator_validation,
    #                    training_particle_IDs, training_labels_particle_IDS,
    #                    validation_particle_IDs, validation_labels_particle_IDS,
    #                    scaler_training_set, path_model)

    for epoch in ["35", "50"]:
        model_epoch = load_model(path_model + "model/weights." + epoch + ".hdf5",
                           custom_objects={'cauchy_selection_loss':lf.cauchy_selection_loss})

        predict_from_model(model_epoch, epoch,
                           generator_training, generator_validation,
                           training_particle_IDs, training_labels_particle_IDS,
                       validation_particle_IDs, validation_labels_particle_IDS,
                       scaler_training_set, path_model)
        del model_epoch
        gc.collect()

############# PLOT ##############

# p_tr10 = np.load("../mse/predicted_training_95.npy")
# t_tr10 = np.load("../mse/true_training_95.npy")
#
# p10 = np.load("../mse/predicted_val_95.npy")
# t10 = np.load("../mse/true_val_95.npy")
#
# p35 = np.load("predicted_val_35.npy")
# t35 = np.load("true_val_35.npy")
#
# p20 = np.load("predicted_val_20.npy")
# t20 = np.load("true_val_20.npy")
#
# p_tr35 = np.load("predicted_training_35.npy")
# t_tr35 = np.load("true_training_35.npy")
#
# p_tr20 = np.load("predicted_training_20.npy")
# t_tr20 = np.load("true_training_20.npy")
#
# f, axes = plt.subplots(3, 2, figsize=(12, 7), sharey=True, sharex=True)
#
# axes[0,0].scatter(t_tr10, p_tr10, s=0.1, label="training set, epoch 10, MSE=%.3f" % mse(t_tr10, p_tr10))
# axes[0, 1].scatter(t10, p10, s=0.2, label="validation set, epoch 10, MSE=%.3f" % mse(t10, p10))
#
# axes[1,0].scatter(t_tr20, p_tr20, s=0.1, label="training set, epoch 20, MSE=%.3f" % mse(t_tr20, p_tr20))
# axes[1, 1].scatter(t20, p20, s=0.2, label="validation set, epoch 20, MSE=%.3f" % mse(t20, p20))
#
# axes[2,0].scatter(t_tr35, p_tr35, s=0.1, label="training set, epoch 35, MSE=%.3f" % mse(t_tr35, p_tr35))
# axes[2,1].scatter(t35, p35, s=0.2, label="validation set, epoch 35, MSE=%.3f" % mse(t35, p35))
#
# for ax in axes.flatten():
#     ax.plot([t_tr20.min(), t_tr20.max()], [t_tr20.min(), t_tr20.max()], color="grey")
#     ax.legend(loc="best", fontsize=13)
#
# plt.subplots_adjust(left=0.08, bottom=0.14, wspace=0, hspace=0)
# plt.ylim(10, 15)
