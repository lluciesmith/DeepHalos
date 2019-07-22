import numpy as np
import data_processing as dp
import CNN
import evaluation
from tensorflow import set_random_seed


if __name__ == "__main__":

    ######### DATA PROCESSING ###########

    p_ids = np.load("/Users/lls/Documents/deep_halos_files/particles.npy")
    p_inputs = np.load("/Users/lls/Documents/deep_halos_files/3d_inputs_particles.npy")
    m = np.load("/Users/lls/Documents/deep_halos_files/outputs_particles.npy")
    scaler, normalised_mass = dp.normalise_output(m, take_log=False)

    # Load the first 500 as training data and the remanining 155 as testing_data
    training_idx = np.arange(500)
    validation_idx = np.arange(500, 654)

    partition = {'train': training_idx, 'validation': validation_idx}

    params_training = {'dim': (17, 17, 17), 'batch_size': 100, 'n_channels': 1, 'shuffle': True}
    params_validation = {'dim': (17, 17, 17), 'batch_size': int(len(validation_idx)/2), 'n_channels': 1,
                         'shuffle': True}

    training_generator = dp.DataGenerator(partition['train'], normalised_mass, **params_training)
    validation_generator = dp.DataGenerator(partition['validation'], normalised_mass, **params_validation)

    ######### TRAINING MODEL ##############

    set_random_seed(7)
    param_conv = {'conv_1': {'num_kernels': 5, 'dim_kernel': (5, 5, 5), 'strides': 1, 'padding': 'valid',
                             'pool': False, 'bn': True},
                  'conv_2': {'num_kernels': 6, 'dim_kernel': (4, 4, 4), 'strides': 1, 'padding': 'valid',
                             'pool': False, 'bn': True},
                  'conv_3': {'num_kernels': 10, 'dim_kernel': (2, 2, 2), 'strides': 1, 'padding': 'valid',
                             'pool': False, 'bn': True}}

    param_fcc = {'dense_1': {'neurons': 80, 'dropout': 0.5}}
    Model = CNN.CNN(training_generator, param_conv, param_fcc, validation_generator, num_epochs=10)
    model = Model.model
    history = Model.history

    ########## PREDICT AND SAVE ############

    pred_cnn_training = model.predict_generator(training_generator)
    training_log_mass = scaler.inverse_transform(pred_cnn_training).flatten()
    # np.save("/share/data2/lls/deep_halos/predicted_log_mass_training.npy", training_log_mass)

    pred_cnn_val = model.predict_generator(validation_generator)
    val_log_mass = scaler.inverse_transform(pred_cnn_val).flatten()
    #np.save("/share/data2/lls/deep_halos/predicted_log_mass_validation.npy", val_log_mass)

    evaluation.plot_loss(history)
    evaluation.plot_metric(history)
    # f = open("/share/data2/lls/deep_halos/history_model.pkl", "wb")
    # pickle.dump(Model.history, f)
    # f.close()

