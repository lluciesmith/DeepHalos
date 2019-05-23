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
    p_ids_saved = np.load("/share/data2/lls/deep_halos/subboxes/ids_saved.npy")

    ######### DATA PROCESSING ###########

    training_ids = np.random.choice(p_ids_saved, 10000, replace=False)
    np.save("/share/data2/lls/deep_halos/training_ids.npy", training_ids)
    validation_ids = np.random.choice(p_ids_saved, 5000, replace=False)
    np.save("/share/data2/lls/deep_halos/validation_ids.npy", validation_ids)

    # for Data generators need ids to be strings and

    partition = {'train': list(training_ids.astype("str")), 'validation': list(validation_ids.astype("str"))}
    labels_dic = dict(zip(list(np.arange(len(normalised_mass)).astype("str")), normalised_mass))

    gen_params = {'dim': (51, 51, 51), 'batch_size': 100, 'n_channels': 1, 'shuffle': True}
    training_generator = dp.DataGenerator(partition['train'], labels_dic, **gen_params)
    validation_generator = dp.DataGenerator(partition['validation'], labels_dic, **gen_params)

    ######### TRAINING MODEL ##############

    set_random_seed(7)
    param_conv = {'conv_1': {'num_kernels': 2, 'dim_kernel': (3, 3, 3), 'strides': 3, 'padding':'valid',
                             'pool': True, 'bn': True},
                  'conv_2': {'num_kernels': 4, 'dim_kernel': (2, 2, 2), 'strides': 2, 'padding':'valid',
                             'pool': False, 'bn': True},
                  # 'conv_3': {'num_kernels': 32, 'dim_kernel': (2, 2, 2), 'strides': 1, 'padding':'valid',
                  #            'pool': False, 'bn': True},
                  # 'conv_4': {'num_kernels': 64, 'dim_kernel': (2, 2, 2), 'strides': 2, 'padding':'valid',
                  #            'pool': False, 'bn': True},
                  # 'conv_5': {'num_kernels': 100, 'dim_kernel': (3, 3, 3), 'strides': 1, 'padding':'valid',
                  #            'pool': False, 'bn': True}
                  # 'conv_6': {'num_kernels': 128, 'dim_kernel': (2, 2, 2), 'strides': 1, 'padding':'valid',
                  #            'pool': False, 'bn': True}
                  }

    param_fcc = {'dense_1': {'neurons': 256, 'dropout': 0.5},
                 'dense_2': {'neurons': 128, 'dropout': 0.5}}

    Model = CNN.CNN(training_generator, validation_generator, param_conv, param_fcc, num_epochs=10,
                    use_multiprocessing=True, workers=24)
    model = Model.model
    history = Model.history

    ########## PREDICT AND SAVE ############

    pred_cnn_training = model.predict_generator(training_generator)
    training_log_mass = scaler.inverse_transform(pred_cnn_training).flatten()
    np.save("/share/data2/lls/deep_halos/predicted_log_mass_training.npy", training_log_mass)
    np.save("/share/data2/lls/deep_halos/true_log_mass_training.npy", np.log10(halo_mass[training_ids]))

    pred_cnn_val = model.predict_generator(validation_generator)
    val_log_mass = scaler.inverse_transform(pred_cnn_val).flatten()
    np.save("/share/data2/lls/deep_halos/predicted_log_mass_validation.npy", val_log_mass)
    np.save("/share/data2/lls/deep_halos/true_log_mass_validation.npy", np.log10(halo_mass[validation_ids]))

    f = open("/share/data2/lls/deep_halos/history_model.pkl", "wb")
    pickle.dump(Model.history, f)
    f.close()

    #
    # cd /share/data2/lls/deep_halos/
    # scp true_log_mass_validation.npy predicted_log_mass_validation.npy lls@star.ucl.ac.uk:/home/lls/