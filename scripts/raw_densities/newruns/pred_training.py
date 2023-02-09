from dlhalos_code_tf2 import CNN
import dlhalos_code_tf2.data_processing as tn
import numpy as np
import importlib
import sys
import pandas as pd

if __name__ == "__main__":
    try:
        params = importlib.import_module(sys.argv[1])
        print("Importin selected param file")
    except IndexError:
        print("Importing default param file")
        import params_raw as params
    print(params.log_alpha)

    # Create the generators for training

    # Create the generators for training
    s = tn.SimulationPreparation(params.all_sims, path=params.path_sims)
    generator_training = tn.DataGenerator(params.training_particle_IDs, params.training_labels_particle_IDS, s.sims_dic,
                                          shuffle=False, path=params.path_data,
                                          cache_path=params.path_data + "raw_tset", **params.params_tr)
    tset = generator_training.get_dataset()

    # Load the model
    for num_epoch_testing in [3, 11]:
        Model = CNN.CNNGaussian(params.param_conv, params.param_fcc,
                                initial_epoch=0, training_generator={}, validation_generator={}, num_epochs=20,
                                dim=generator_training.dim, initialiser="Xavier_uniform", verbose=1, num_gpu=1,
                                lr=params.lr, save_summary=True, path_summary=params.saving_path, train=False,
                                compile=True, seed=params.seed)
        Model.model.load_weights(params.saving_path + "model/weights.%02d.h5" % num_epoch_testing)

        # Predict
        pred = params.scaler.inverse_transform(Model.model.predict(tset, verbose=1).reshape(-1, 1)).flatten()
        true = params.scaler.inverse_transform(np.concatenate([y for x, y in tset], axis=0).reshape(-1, 1)).flatten()
        np.save(params.saving_path + "predicted_sim_trainingset_epoch_%02d.npy" % num_epoch_testing, pred)
        np.save(params.saving_path + "true_sim_trainingset_epoch_%02d.npy" % num_epoch_testing, true)
        del Model


