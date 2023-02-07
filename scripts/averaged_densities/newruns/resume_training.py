from dlhalos_code_tf2 import CNN
import dlhalos_code_tf2.data_processing as tn
import numpy as np
import importlib
import sys

if __name__ == "__main__":
    try: params = importlib.import_module(sys.argv[1])
    except IndexError: import params_avg as params
    print(params.log_alpha)

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    s = tn.SimulationPreparation(params.all_sims, path=params.path_sims)

    # Create the generators for training

    generator_training = tn.DataGenerator(params.training_particle_IDs, params.training_labels_particle_IDS, s.sims_dic, shuffle=False, path=params.path_data, cache_path=params.path_data + "tset", **params.params_tr, **params.params_box)
    generator_validation = tn.DataGenerator(params.val_particle_IDs, params.val_labels_particle_IDS, s.sims_dic, shuffle=False, path=params.path_data, cache_path=params.path_data + "vset", **params.params_val, **params.params_box)
    tset = generator_training.get_dataset()
    vset = generator_validation.get_dataset()

    ######### TRAIN THE MODEL ################

    Model = CNN.CNNGaussian(params.param_conv, params.param_fcc,
                            initial_epoch=0,training_generator=tset, steps_per_epoch=len(generator_training),
                            validation_generator=vset, num_epochs=20, dim=generator_training.dim,
                            initialiser="Xavier_uniform", verbose=1, num_gpu=1, lr=params.lr,
                            save_summary=True, path_summary=params.saving_path, train=False, compile=True, seed=params.seed)

    # test it on the validation set

    tr = np.loadtxt(params.saving_path + "training.log", delimiter=",", skiprows=1)
    num_epoch_testing = np.max(tr[:, 0]) + 1
    weights = params.saving_path + "model/weights.%02d.h5" % num_epoch_testing
    Model.model.load_weights(weights)
    Model.model.fit(tset, validation_data=vset, initial_epoch=int(num_epoch_testing), epochs=20, callbacks=Model.callbacks)