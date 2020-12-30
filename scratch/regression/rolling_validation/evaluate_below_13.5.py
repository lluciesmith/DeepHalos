import sys
sys.path.append("/home/luisals/DeepHalos")
from tensorflow.keras.models import load_model
import dlhalos_code.data_processing as tn
import numpy as np
from pickle import load

if __name__ == "__main__":
    ########## CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # First choose the correct path to the model and the parameters you used during training
    # params_model0 = {'path_model': "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval_0dropout/"}
    params_model01 = {'path_model': "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval_0.1dropout/"}
    params_model02 = {'path_model': "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval_0.2dropout/"}
    params_model03 = {'path_model': "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3_w_eval_0.3dropout/"}
    params_model04 = {'path_model': "/lfstev/deepskies/luisals/regression/rolling_val/no_sim3/"}

    for params_model in [ params_model01, params_model02, params_model03, params_model04]:

        params_inputs = {'batch_size': 80,
                         'rescale_mean': 1.005,
                         'rescale_std': 0.05050,
                         'dim': (75, 75, 75)
                         }
        s_output = load(open(params_model['path_model'] + 'scaler_output.pkl', 'rb'))

        epochs = [5 * i for i in range(1, 21)]
        epochs = np.array(epochs).astype('str')
        epochs[0] = "05"

        loss = np.zeros((len(epochs), 4))
        loss[:, 0] = [5 * i for i in range(1, 21)]

        #################### training set ####################

        all_sims = ["0", "1", "2", "4", "5", "6"]
        s = tn.SimulationPreparation(all_sims)
        training_set = tn.InputsPreparation(all_sims, load_ids=True, random_subset_all=4000, log_high_mass_limit=13.5,
                                            scaler_output=s_output, shuffle=False)
        generator_train = tn.DataGenerator(training_set.particle_IDs, training_set.labels_particle_IDS, s.sims_dic,
                                           **params_inputs)

        for i, epoch in enumerate(epochs):
            model_epoch = load_model(params_model['path_model'] + "model/weights." + epoch + ".hdf5")
            loss[i, 1] = model_epoch.evaluate_generator(generator_train, use_multiprocessing=False, workers=1,
                                                        verbose=1)
            del model_epoch

        #################### simulation-7 ####################

        val_sim = ["7"]
        s_val = tn.SimulationPreparation(val_sim)
        validation_set7 = tn.InputsPreparation(val_sim, load_ids=True, random_subset_each_sim=4000,
                                               log_high_mass_limit=13.5, scaler_output=s_output, shuffle=False)
        generator7 = tn.DataGenerator(validation_set7.particle_IDs, validation_set7.labels_particle_IDS,
                                                s_val.sims_dic, **params_inputs)

        for i, epoch in enumerate(epochs):
            model_epoch = load_model(params_model['path_model'] + "model/weights." + epoch + ".hdf5")
            loss[i, 2] = model_epoch.evaluate_generator(generator7, use_multiprocessing=False, workers=1, verbose=1)
            del model_epoch

        #################### validation sim at given epoch ####################

        validation_sims_each_epoch = np.concatenate(np.load(params_model['path_model'] + 'validation_sims.npy'))

        for i, epoch in enumerate(epochs):
            epoch_int = epoch.astype("int")
            val_sim_epoch = validation_sims_each_epoch[epoch_int - 1]
            s1 = tn.SimulationPreparation(val_sim_epoch)
            val_set = tn.InputsPreparation(val_sim_epoch, load_ids=True, random_subset_each_sim=4000,
                                           log_high_mass_limit=13.5, scaler_output=s_output, shuffle=False)
            generator_val = tn.DataGenerator(val_set.particle_IDs, val_set.labels_particle_IDS, s1.sims_dic,
                                             **params_inputs)
            model_epoch = load_model(params_model['path_model'] + "model/weights." + epoch + ".hdf5")
            loss[i, 3] = model_epoch.evaluate_generator(generator_val, use_multiprocessing=False, workers=1, verbose=1)
            del model_epoch

        np.save(params_model['path_model'] + "loss_training_sim7_validaton_above_135Msol.npy", loss)
