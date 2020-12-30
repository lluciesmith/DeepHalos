from dlhalos_code import CNN
import dlhalos_code.z0_data_processing as dp0
import importlib
import sys

if __name__ == "__main__":
    try: params = importlib.import_module(sys.argv[1])
    except IndexError: import params_z0 as params

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # Create the generators for training
    s = dp0.SimulationPreparation_z0(params.all_sims, path=params.path_sims)

    generator_training = dp0.DataGenerator_z0(params.training_particle_IDs, params.training_labels_particle_IDS,
                                              s.sims_dic, shuffle=True, batch_size=64, **params.params_box)
    generator_validation = dp0.DataGenerator_z0(params.val_particle_IDs, params.val_labels_particle_IDS, s.sims_dic,
                                                gridded_box=generator_training.box_class, shuffle=False, batch_size=50,
                                                **params.params_box)

    ######### TRAIN THE MODEL ################

    Model = CNN.CNNCauchy(params.param_conv, params.param_fcc, model_type="regression",
                          training_generator=generator_training, shuffle=True,
                          validation_generator=generator_validation, num_epochs=30,
                          metrics=[CNN.likelihood_metric],
                          steps_per_epoch=len(generator_training), validation_steps=len(generator_validation),
                          dim=generator_training.dim, initialiser="Xavier_uniform", max_queue_size=8,
                          use_multiprocessing=False, workers=0, verbose=1, num_gpu=1, lr=params.lr, save_summary=True,
                          path_summary=params.saving_path, validation_freq=1, train=True, compile=True,
                          initial_epoch=None, lr_scheduler=False, seed=params.seed)

