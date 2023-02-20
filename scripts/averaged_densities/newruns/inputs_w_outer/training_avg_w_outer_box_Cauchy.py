from dlhalos_code_tf2 import CNN
import dlhalos_code_tf2.data_processing as tn
import numpy as np
import importlib
import sys
import pandas as pd

if __name__ == "__main__":
    try: params = importlib.import_module(sys.argv[1])
    except IndexError: import params_avg_outershell as params
    print(params.log_alpha)

    cache_path = '/share/data2/lls/'

    # Create the generators for training
    s = tn.SimulationPreparation(params.all_sims, path=params.path_sims)
    generator_training = tn.DataGenerator(params.training_particle_IDs, params.training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, path=params.path_data, cache_path=cache_path + "outershell_tset",
                                          **params.params_tr, **params.params_box)
    generator_validation = tn.DataGenerator(params.val_particle_IDs, params.val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, path=params.path_data, cache_path=cache_path + "outershell_vset",
                                            **params.params_val, **params.params_box)
    tset = generator_training.get_dataset()
    vset = generator_validation.get_dataset()

    # Train the model
    Model = CNN.CNNCauchy(params.param_conv, params.param_fcc, train_gamma=False, init_gamma=0.2,
                          initial_epoch=0, training_dataset=tset, steps_per_epoch=len(generator_training),
                          validation_steps=len(generator_validation),
                          validation_dataset=vset, num_epochs=20, dim=generator_training.dim, metrics=[CNN.likelihood_metric],
                          initialiser="Xavier_uniform", verbose=1, num_gpu=1, lr=params.lr, seed=params.seed,
                          save_summary=True, path_summary=params.saving_path, train=True, compile=True)

    # Test the model
    tr = pd.read_csv(params.saving_path + 'training.log', sep=",", header=0)
    num_epoch_testing = np.argmin(tr['val_loss']) + 1
    weights = params.saving_path + "model/weights.%02d.h5" % num_epoch_testing
    Model.model.load_weights(weights)

    for name, dset in [("valset", vset), ("tset", tset)]:
        pred = params.scaler.inverse_transform(Model.model.predict(dset, verbose=1).reshape(-1, 1)).flatten()
        true = params.scaler.inverse_transform(np.concatenate([y for x, y in dset], axis=0).reshape(-1, 1)).flatten()
        np.save(params.saving_path + "predicted_sim_" + name + "_epoch_%02d.npy" % num_epoch_testing, pred)
        np.save(params.saving_path + "true_sim_" + name + "_epoch_%02d.npy" % num_epoch_testing, true)

