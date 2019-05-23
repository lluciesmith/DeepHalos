import numpy as np
import sys; sys.path.append("/home/lls/DeepHalos/")
# import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
# sys.path.append("/Users/lls/Documents/Projects/")
import data_processing as dp
import CNN
from tensorflow import set_random_seed
import pickle


if __name__ == "__main__":
    path = "/home/lls/stored_files"
    # saving_path = "/share/data2/lls/deep_halos/subboxes"

    halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
    scaler, normalised_mass = dp.normalise_output(halo_mass, take_log=True)
    p_ids = np.where(halo_mass > 0)[0]

    ######### DATA PROCESSING ###########

    # Load the first 50000 as training data and the next 10000 as validation data
    num_training_ids = 50000
    num_valid_ids = 10000
    training_ids = p_ids[:num_training_ids]
    validation_ids = p_ids[num_training_ids: num_training_ids + num_valid_ids]

    partition = {'train': training_ids, 'validation': validation_ids}
    gen_params = {'dim': (51, 51, 51), 'batch_size': 1000, 'n_channels': 1, 'shuffle': True}

    training_generator = dp.DataGenerator(partition['train'], normalised_mass, **gen_params)
    validation_generator = dp.DataGenerator(partition['validation'], normalised_mass, **gen_params)

    ######### TRAINING MODEL ##############

    set_random_seed(7)
    param_conv = {'conv_1': {'num_kernels': 2, 'dim_kernel': (48, 48, 48), 'strides': 1, 'padding':'valid',
                             'pool': True, 'bn': True},
                  'conv_2': {'num_kernels': 12, 'dim_kernel': (22, 22, 22), 'strides': 1, 'padding':'valid',
                             'pool': True, 'bn': True},
                  'conv_3': {'num_kernels': 32, 'dim_kernel': (9, 9, 9), 'strides': 1, 'padding':'valid',
                             'pool': False, 'bn': True},
                  'conv_4': {'num_kernels': 64, 'dim_kernel': (6, 6, 6), 'strides': 1, 'padding':'valid',
                             'pool': False, 'bn': True},
                  'conv_5': {'num_kernels': 128, 'dim_kernel': (4, 4, 4), 'strides': 1, 'padding':'valid',
                             'pool': False, 'bn': True},
                  'conv_6': {'num_kernels': 128, 'dim_kernel': (2, 2, 2), 'strides': 1, 'padding':'valid',
                             'pool': False, 'bn': True}}

    param_fcc = {'dense_1': {'neurons': 1024, 'dropout': 0.5},
                 'dense_2': {'neurons': 256, 'dropout': 0.5}}

    Model = CNN.CNN(training_generator, validation_generator, param_conv, param_fcc, num_epochs=100,
                    use_multiprocessing=True, workers=24)
    model = Model.model
    history = Model.history

    ########## PREDICT AND SAVE ############

    pred_cnn_training = model.predict_generator(training_generator)
    training_log_mass = scaler.inverse_transform(pred_cnn_training).flatten()
    np.save("/share/data2/lls/deep_halos/predicted_log_mass_training.npy", training_log_mass)

    pred_cnn_val = model.predict_generator(validation_generator)
    val_log_mass = scaler.inverse_transform(pred_cnn_val).flatten()
    np.save("/share/data2/lls/deep_halos/predicted_log_mass_validation.npy", val_log_mass)

    f = open("/share/data2/lls/deep_halos/history_model.pkl", "wb")
    pickle.dump(Model.history, f)
    f.close()



