import numpy as np
import sys; sys.path.append("/home/lls/DeepHalos/")
# import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
# sys.path.append("/Users/lls/Documents/Projects/")
import data_processing as dp
import CNN
from tensorflow import set_random_seed
import sklearn.preprocessing
import pickle


def normalise_output(halo_mass):
    log_mass = np.log10(halo_mass[halo_mass > 0])
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    minmax_scaler.fit(log_mass.reshape(-1, 1))

    normalised_labels = np.zeros((len(halo_mass),))
    normalised_labels[halo_mass > 0] = minmax_scaler.transform(log_mass.reshape(-1, 1)).flatten()
    return minmax_scaler, normalised_labels

# def load_data(particles, shape_subboxes=51, saving_path="/share/data2/lls/deep_halos/subboxes"):
#     f1 = np.zeros((len(particles), shape_subboxes, shape_subboxes, shape_subboxes))
#
#     for i in range(len(particles)):
#         particle_id = particles[i]
#         if shape_subboxes == 51:
#             f1[i] = np.load(saving_path + "/subbox_51_particle_" + str(particle_id) + ".npy")
#     return f1


if __name__ == "__main__":
    path = "/home/lls/stored_files"
    # saving_path = "/share/data2/lls/deep_halos/subboxes"

    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
    scaler, normalised_mass = normalise_output(halo_mass)
    p_ids = np.where(halo_mass > 0)[0]

    ######### DATA PROCESSING ###########

    # Load the first 5000 as training data and the next 2000 as testing_data
    num_training_ids = 50000
    num_valid_ids = 10000
    training_ids = p_ids[:num_training_ids]
    validation_ids = p_ids[num_training_ids: num_training_ids + num_valid_ids]

    partition = {'train': training_ids,
                 'validation': validation_ids}

    params = {'dim': (51, 51, 51),
              'batch_size': 1000,
              'n_channels': 1,
              'shuffle': True}

    training_generator = dp.DataGenerator(partition['train'], normalised_mass, **params)
    validation_generator = dp.DataGenerator(partition['validation'], normalised_mass, **params)

    ######### TRAINING MODEL ##############

    set_random_seed(7)

    model_params = {'num_convolutions': 6,
                    'num_kernels': [2, 12, 32, 64, 128, 128],
                    'dim_kernel': [(48, 48, 48), (22, 22, 22), (9, 9, 9), (6, 6, 6), (4, 4, 4), (2, 2, 2)],
                    # 'strides': [2, 2, 2, 2, 2, 2],
                    'strides': [1, 1, 1, 1, 1, 1],
                    'padding':'valid',
                    'dense_neurons': [1024, 256],
                    'pool': [True, True, False, False, False, False]
                    }

    Model = CNN.model_w_layers(input_shape_box=(51, 51, 51), **model_params)
    history = Model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=True, workers=24,
                                  verbose=1, epochs=100, shuffle=True)

    ########## PREDICT AND SAVE ############

    pred_cnn_training = Model.predict_generator(training_generator)
    training_log_mass = scaler.inverse_transform(pred_cnn_training).flatten()
    np.save("/share/data2/lls/deep_halos/predicted_log_mass_training.npy", training_log_mass)

    pred_cnn_val = Model.predict_generator(validation_generator)
    val_log_mass = scaler.inverse_transform(pred_cnn_val).flatten()
    np.save("/share/data2/lls/deep_halos/predicted_log_mass_validation.npy", val_log_mass)

    f = open("/share/data2/lls/deep_halos/history_model.pkl", "wb")
    pickle.dump(Model.history, f)
    f.close()



