import sys
sys.path.append("/home/luisals/DeepHalos")
from dlhalos_code import CNN
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import CSVLogger
import tensorflow
from tensorflow.keras.models import load_model
import dlhalos_code.data_processing as tn
import numpy as np
from pickle import dump
import time


def split_training_set_into_train_val_sims(sims_prep, output_scaler, batch_size=80,
                                   rescale_mean=1.005, rescale_std=0.05050, dim=(121, 121, 121)):
    all_sims = list(np.copy(sims_prep.sims))
    np.random.seed()
    n = int(np.random.choice(np.arange(len(all_sims)), 1))
    val_sim = list(all_sims.pop(n))
    train_sims = all_sims
    training_set = tn.InputsPreparation(train_sims, load_ids=True, scaler_output=output_scaler, shuffle=True)
    generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS,
                                          sims_prep.sims_dic, batch_size=batch_size, rescale_mean=rescale_mean,
                                          rescale_std=rescale_std, dim=dim)
    training_set_val = tn.InputsPreparation(train_sims, load_ids=True, random_subset_all=4000,
                                            scaler_output=output_scaler, shuffle=True)
    generator_val_training = tn.DataGenerator(training_set_val.particle_IDs, training_set_val.labels_particle_IDS,
                                          sims_prep.sims_dic, batch_size=batch_size, rescale_mean=rescale_mean,
                                          rescale_std=rescale_std, dim=dim)
    return generator_training, generator_val_training


def split_training_validation_sims(sims_prep, output_scaler, batch_size=80,
                                   rescale_mean=1.005, rescale_std=0.05050, dim=(121, 121, 121)):
    all_sims = list(np.copy(sims_prep.sims))
    np.random.seed()
    n = int(np.random.choice(np.arange(len(all_sims)), 1))
    val_sim = list(all_sims.pop(n))
    train_sims = all_sims
    print(val_sim)
    training_set = tn.InputsPreparation(train_sims, load_ids=True, scaler_output=output_scaler, shuffle=True)
    generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS,
                                          sims_prep.sims_dic, batch_size=batch_size, rescale_mean=rescale_mean,
                                          rescale_std=rescale_std, dim=dim)
    validation_set = tn.InputsPreparation(val_sim, load_ids=True, random_subset_each_sim=4000,
                                          scaler_output=output_scaler)
    generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS,
                                            sims_prep.sims_dic, batch_size=batch_size, rescale_mean=rescale_mean,
                                            rescale_std=rescale_std, dim=dim)
    print("Done data preparation")
    return generator_training, generator_validation, val_sim


if __name__ == "__main__":

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    path_model = "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval_0.1dropout/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    # define a common scaler for the output

    data_set = tn.InputsPreparation(all_sims, load_ids=True)
    scaler_output = data_set.scaler_output
    dump(scaler_output, open(path_model + 'scaler_output.pkl', 'wb'))


    ######### TRAINING MODEL ##############

    # checkpoint
    filepath = path_model + "/model/weights.{epoch:02d}.hdf5"
    checkpoint_call = callbacks.ModelCheckpoint(filepath, period=5)

    # save histories
    csv_logger = CSVLogger(path_model + "/training.log", separator=',', append=True)

    callbacks_list = [checkpoint_call, csv_logger]
    tensorflow.compat.v1.set_random_seed(7)

    param_conv = {'conv_1': {'num_kernels': 4, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  'conv_2': {'num_kernels': 8, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  'conv_3': {'num_kernels': 16, 'dim_kernel': (3, 3, 3),
                             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  # 'conv_4': {'num_kernels': 32, 'dim_kernel': (3, 3, 3),
                  #             'strides': 1, 'padding': 'same', 'pool': "max", 'bn': True},
                  'conv_4': {'num_kernels': 4, 'dim_kernel': (1, 1, 1),
                             'strides': 1, 'padding': 'same', 'pool': None, 'bn': True}
                  }

    param_fcc = {'dense_1': {'neurons': 256, 'bn': True, 'dropout': 0.1},
                 'dense_2': {'neurons': 128, 'bn': False, 'dropout': 0.1}}

    Model = CNN.CNN(param_conv, param_fcc, model_type="regression", training_generator={}, validation_generator={},
                    callbacks=callbacks_list, num_epochs=100, dim=(75, 75, 75), max_queue_size=10,
                    use_multiprocessing=True, workers=2, verbose=1, lr=0.0001, validation_freq=1, train=False)

    model = Model.model
    epochs = Model.num_epochs

    # model = load_model("/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval/model/weights.45.hdf5")
    # epochs_init = 45
    val_sims = []

    for epoch in np.arange(epochs):
        train_gen, val_gen, val_sim = split_training_validation_sims(s, scaler_output, batch_size=80,
                                                                                rescale_mean=1.005,
                                                                                rescale_std=0.05050, dim=(75, 75, 75))
        val_sims.append(val_sim)
        # train_gen, val_gen = split_training_set_into_train_val_sims(s, scaler_output, batch_size=80, rescale_mean=1.005,
        #                                                     rescale_std=0.05050, dim=(75, 75, 75))
        history = model.fit_generator(generator=train_gen, validation_data=val_gen,
                                      use_multiprocessing=True, workers=2, max_queue_size=10, verbose=1,
                                      epochs=epoch+1, shuffle=True, callbacks=callbacks_list, validation_freq=1,
                                      initial_epoch=epoch)

    model.save(path_model + "/model_100_epochs_mixed_sims.h5")
    np.save(path_model + "/validation_sims.npy", val_sims)


# sims = ["0", "1", "2", "4", "5", "6"]
# h = []
# for sim in sims:
#     if sim == "0":
#         hi = np.load("/lfstev/deepskies/luisals/training_simulation/halo_mass_particles.npy")
#         ids = np.loadtxt("/lfstev/deepskies/luisals/training_simulation/training_sim_random_training_set.txt",
#                          delimiter=",", dtype="int")
#         h.append(np.log10(hi[ids]))
#     else:
#         hi = np.load("/lfstev/deepskies/luisals/reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")
#         ids = np.loadtxt("/lfstev/deepskies/luisals/reseed" + sim + "_simulation/reseed_" + sim +
#                       "_random_training_set.txt", delimiter=",", dtype="int")
#         h.append(np.log10(hi[ids]))