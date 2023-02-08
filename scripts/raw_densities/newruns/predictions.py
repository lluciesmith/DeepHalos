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

    s = tn.SimulationPreparation(params.test_sim, path=params.path_sims)
    generator_test = tn.DataGenerator(params.test_particle_IDs, params.test_labels_particle_IDS, s.sims_dic,
                                      cache_path=params.path_data + "testset",
                                      shuffle=False, path=params.path_data, **params.params_val, **params.params_box)
    testset = generator_test.get_dataset()

    # Load the model
    tr = pd.read_csv(params.saving_path + 'training.log', sep=",", header=0)
    num_epoch_testing = np.argmin(tr['val_loss']) + 1
    Model = CNN.CNNGaussian(params.param_conv, params.param_fcc,
                            initial_epoch=0, training_generator={}, validation_generator={}, num_epochs=20,
                            dim=generator_test.dim, initialiser="Xavier_uniform", verbose=1, num_gpu=1, lr=params.lr,
                            save_summary=True, path_summary=params.saving_path, train=False, compile=True, seed=params.seed)
    Model.model.load_weights(params.saving_path + "model/weights.%02d.h5" % num_epoch_testing)

    # Predict
    pred = params.scaler.inverse_transform(Model.model.predict(testset, verbose=1).reshape(-1, 1)).flatten()
    true = params.scaler.inverse_transform(np.concatenate([y for x, y in testset], axis=0).reshape(-1, 1)).flatten()
    np.save(params.saving_path + "predicted_sim_testset_epoch_%02d.npy" % num_epoch_testing, pred)
    np.save(params.saving_path + "true_sim_testset_epoch_%02d.npy" % num_epoch_testing, true)


