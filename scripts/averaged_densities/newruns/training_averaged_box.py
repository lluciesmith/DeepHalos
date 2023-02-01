from dlhalos_code import CNN
import dlhalos_code.data_processing as tn

if __name__ == "__main__":
    import params_avg as params

    ########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    s = tn.SimulationPreparation(params.all_sims, path=params.path_sims)

    # Create the generators for training

    generator_training = tn.DataGenerator(params.training_particle_IDs, params.training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params.params_tr, **params.params_box)
    generator_validation = tn.DataGenerator(params.val_particle_IDs, params.val_labels_particle_IDS, s.sims_dic,
                                            shuffle=False, **params.params_val, **params.params_box)

    ######### TRAIN THE MODEL ################

    Model = CNN.CNNGaussian(params.param_conv, params.param_fcc,
                            training_generator=generator_training, steps_per_epoch=len(generator_training),
                            validation_generator=generator_validation, num_epochs=20, dim=generator_training.dim,
                            initialiser="Xavier_uniform", max_queue_size=8, use_multiprocessing=False, workers=0,
                            verbose=1, num_gpu=1, lr=params.lr, save_summary=True, path_summary=params.saving_path,
                            train=True, compile=True, initial_epoch=0, lr_scheduler=False, seed=params.seed)

