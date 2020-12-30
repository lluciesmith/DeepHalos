from dlhalos_code import CNN
import dlhalos_code.data_processing as tn
import numpy as np
import importlib
import sys

if __name__ == "__main__":
    try:
        params = importlib.import_module(sys.argv[1])
    except IndexError:
        import params_avg as params

    # Load data

    s = tn.SimulationPreparation([params.val_sim], path=params.path_sims)
    generator_validation = tn.DataGenerator(params.large_val_particle_IDs, params.large_val_labels_particle_IDS,
                                            s.sims_dic,
                                            shuffle=False, **params.params_val, **params.params_box)

    # Model

    Model = CNN.CNNCauchy(params.param_conv, params.param_fcc, model_type="regression", training_generator={},
                          shuffle=True,
                          validation_generator=generator_validation, num_epochs=100, dim=generator_validation.dim,
                          max_queue_size=80, use_multiprocessing=False, workers=0, verbose=1, num_gpu=1,
                          lr=params.lr,
                          save_summary=False, path_summary=params.saving_path, validation_freq=1, train=False,
                          compile=True,
                          initial_epoch=None, initialiser="Xavier_uniform")

    epochs = ["%02d" % num for num in np.arange(1, 41)][10::2]
    loss_val = []

    for num_epoch in epochs:
        weights = params.saving_path + "model/weights." + num_epoch + ".h5"
        Model.model.load_weights(weights)
        l_v = Model.model.evaluate_generator(generator_validation, use_multiprocessing=False, workers=0, verbose=1,
                                             steps=len(generator_validation))
        loss_val.append(l_v)

    np.save(params.saving_path + "loss_larger_validation_set.npy", loss_val)
