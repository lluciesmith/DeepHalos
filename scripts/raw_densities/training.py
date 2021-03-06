from dlhalos_code import CNN
import dlhalos_code.data_processing as tn
import numpy as np
import tensorflow as tf
import random as python_random
import sys
import importlib


if __name__ == "__main__":
    try: params = importlib.import_module(sys.argv[1])
    except IndexError: import params_avg as params

    np.random.seed(params.seed)
    python_random.seed(params.seed)
    tf.compat.v1.set_random_seed(params.seed)

    # Prepare the data

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