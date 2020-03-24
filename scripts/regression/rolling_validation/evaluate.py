import sys
sys.path.append("/home/luisals/DeepHalos")
from tensorflow.keras.models import load_model
import dlhalos_code.data_processing as tn
import numpy as np
from pickle import load


if __name__ == "__main__":
    ########## CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # First choose the correct path to the model and the parameters you used during training
    params_model0 = {'path_model': "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval_0dropout/"}
    params_model01 = {'path_model': "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval_0.1dropout/"}
    params_model02 = {'path_model':"/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval_0.2dropout/"}
    params_model03 = {'path_model': "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval_0.3dropout/"}
    params_model04 = {'path_model': "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3/"}

    for params_model in [params_model0, params_model01, params_model02, params_model03, params_model04]:

        params_inputs = {'batch_size': 80,
                         'rescale_mean': 1.005,
                         'rescale_std': 0.05050,
                         'dim': (75, 75, 75)
                         }
        s_output = load(open(params_model['path_model'] + 'scaler_output.pkl', 'rb'))

        val_sim = ["7"]
        s_val = tn.SimulationPreparation(val_sim)
        validation_set = tn.InputsPreparation(val_sim, load_ids=True, scaler_output=s_output,
                                              random_subset_each_sim=4000, shuffle=False)
        generator_validation = tn.DataGenerator(validation_set.particle_IDs, validation_set.labels_particle_IDS,
                                                s_val.sims_dic, **params_inputs)

        all_sims = ["0", "1", "2", "4", "5", "6"]
        s = tn.SimulationPreparation(all_sims)
        training_set = tn.InputsPreparation(all_sims, load_ids=True, random_subset_all=4000,
                                            scaler_output=s_output, shuffle=False)
        generator_training = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS,
                                              s.sims_dic, **params_inputs)

        epochs = [5*i for i in range(1, 21)]
        epochs = np.array(epochs).astype('str')
        epochs[0] = "05"

        loss = np.zeros((len(epochs), 3))
        loss[:, 0] = [5*i for i in range(1, 21)]

        for i, epoch in enumerate(epochs):
            model_epoch = load_model(params_model['path_model'] + "model/weights." + epoch + ".hdf5")
            # loss[i, 1] = model_epoch.evaluate_generator(generator_training, use_multiprocessing=False, workers=1,
            #                                             verbose=1)
            loss[i, 2] = model_epoch.evaluate_generator(generator_validation, use_multiprocessing=False, workers=1,
                                                        verbose=1)
            del model_epoch

        np.save(params_model['path_model'] + "loss_training_and_sim7.npy", loss)
