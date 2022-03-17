from dlhalos_code import CNN
import dlhalos_code.data_processing as tn
import numpy as np
import tensorflow as tf
import random as python_random
import sys
import importlib
from pickle import dump
from collections import OrderedDict


def turn_mass_labels_into_ellipticity(labels, scale, path, scaler=None):
    ids_keys = labels.keys()
    sim_index = np.array([ID[ID.find('sim-') + 4: ID.find('-id')] for ID in ids_keys])
    particle_ids = np.array([int(ID[ID.find('-id-') + 4:]) for ID in ids_keys])
    labels_ell = np.zeros((len(particle_ids),))
    sims_unique = np.unique(sim_index)
    for sim in sims_unique[11:]:
        if sim == "0":
            ell = np.load(path + "training_simulation/reseed" + sim + "_densub_ellipticity_scale_%.2f.npy" % float(scale))
        else:
            ell = np.load(path + "reseed" + sim + "_simulation/reseed" + sim + "_densub_ellipticity_scale_%.2f.npy" % float(scale))
        labels_ell[sim_index == sim] = ell[particle_ids[sim_index == sim]]
    if scaler is None:
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(labels_ell.reshape(-1, 1))
        rescaled_labels = scaler.transform(labels_ell.reshape(-1, 1)).flatten()
        return OrderedDict(zip(ids_keys, rescaled_labels)), scaler
    else:
        rescaled_labels = scaler.transform(labels_ell.reshape(-1, 1)).flatten()
        return OrderedDict(zip(ids_keys, rescaled_labels))


if __name__ == "__main__":
    try: params = importlib.import_module(sys.argv[1])
    except IndexError: import params_ell as params

    np.random.seed(params.seed)
    python_random.seed(params.seed)
    tf.compat.v1.set_random_seed(params.seed)

    # Prepare the data
    scale0 = params.smoothing_scales[0]
    labels_tr, scaler_tr = turn_mass_labels_into_ellipticity(params.training_labels_particle_IDS, scale0, params.path_sims)
    params.training_labels_particle_IDS = labels_tr
    with open(params.saving_path + "scaler_output_ell.pickle", "wb") as output_file:
        dump(scaler_tr, output_file)

    labels_val = turn_mass_labels_into_ellipticity(params.val_labels_particle_IDS, scale0, params.path_sims, scaler=scaler_tr)
    params.val_labels_particle_IDS = labels_val

    # typical pipeline
    s = tn.SimulationPreparation(params.all_sims, path=params.path_sims)
    generator_training = tn.DataGenerator(params.training_particle_IDs, params.training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params.params_tr)
    generator_validation = tn.DataGenerator(params.val_particle_IDs, params.val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params.params_val)


    ######### TRAIN THE MODEL ################

    Model = CNN.CNNCauchy(params.param_conv, params.param_fcc, model_type="regression", training_generator=generator_training,
                          shuffle=True, validation_generator=generator_validation, num_epochs=30,
                          metrics=[CNN.likelihood_metric],
                          steps_per_epoch=len(generator_training), validation_steps=len(generator_validation),
                          dim=generator_training.dim, initialiser="Xavier_uniform", max_queue_size=10,
                          use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=params.lr, save_summary=True,
                          path_summary=params.saving_path, validation_freq=1, train=True, compile=True,
                          initial_epoch=None, lr_scheduler=False,
                          seed=params.seed)