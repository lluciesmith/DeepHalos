import sys
sys.path.append("/home/luisals/DeepHalos")
import dlhalos_code.data_processing as tn
from tensorflow.keras.models import load_model
import numpy as np
from pickle import load


def get_sim_predictions_given_model(sims, model, scaler_output, val_sim="1",
                                    path_model=".", batch_size=80, rescale_mean=1.005, rescale_std=0.05050,
                                    dim=(121, 121, 121), save=True, num_epoch="100"):

    validation_set = tn.InputsPreparation([val_sim], load_ids=True, scaler_output=scaler_output, shuffle=False)
    generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS,
                                            sims.sims_dic,
                                            batch_size=batch_size, rescale_mean=rescale_mean, rescale_std=rescale_std,
                                            dim=dim)

    pred = model.predict_generator(generator_validation, use_multiprocessing=False, workers=1, verbose=1)
    truth_rescaled = np.array([val for (key, val) in validation_set.labels_particle_IDS.items()])

    h_m_pred = scaler_output.inverse_transform(pred).flatten()
    true = scaler_output.inverse_transform(truth_rescaled).flatten()

    if save is True:
        np.save(path_model + "predicted" + val_sim + "_" + num_epoch + ".npy", h_m_pred)
        np.save(path_model + "true" + val_sim + "_" + num_epoch + ".npy", true)

    return h_m_pred, true


if __name__ == "__main__":
    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # First choose the correct path to the model and the parameters you used during training

    params_model = {# 'path_model':"/lfstev/deepskies/luisals/regression/rolling_val/",
                    #'path_model': "/lfstev/deepskies/luisals/regression/train_mixed_sims/51_3_maxpool/",
                    'path_model': "/lfstev/deepskies/luisals/regression/ics_res121/stride1/",
                    'num_epoch': "30",
                    'batch_size': 40,
                    'rescale_mean': 1.005,
                    'rescale_std': 0.05050,
                    'dim': (121, 121, 121)
                    }

    # load validation sets

    validation_sims = ["1"]
    s = tn.SimulationPreparation(validation_sims)
    s_output = load(open(params_model['path_model'] + 'scaler_output.pkl', 'rb'))

    # load model

    # model_roll = load_model(params_model['path_model'] + "/model_100_epochs_mixed_sims.h5")
    model_epoch = load_model(params_model['path_model'] + "model/weights." + params_model['num_epoch'] + ".hdf5")

    for val_sim in validation_sims:
        pi, ti = get_sim_predictions_given_model(s, model_epoch, s_output, val_sim="1", **params_model)
        print(np.corrcoef(ti, pi))

