import dlhalos_code.z0_data_processing as dp0
from pickle import load
import numpy as np


if __name__ == "__main__":

########### CREATE GENERATORS FOR TRAINING AND VALIDATION #########

    # Load data

    path_sims = "/mnt/beegfs/work/ati/pearl037/"
    all_sims = ["%i" % i for i in np.arange(22)]
    all_sims.remove("3")
    s = dp0.SimulationPreparation_z0(all_sims, path=path_sims)

    path_data = "/mnt/beegfs/work/ati/pearl037/regression/training_set_13.4/20sims/random/200k/"
    training_particle_IDs = load(open(path_data + 'training_set.pkl', 'rb'))
    training_labels_particle_IDS = load(open(path_data + 'labels_training_set.pkl', 'rb'))
    val_particle_IDs = load(open(path_data + 'validation_set.pkl', 'rb'))
    val_labels_particle_IDS = load(open(path_data + 'labels_validation_set.pkl', 'rb'))

    # Create the generators for training

    dim = (75, 75, 75)
    params_box = {'res_sim': 1667, 'rescale': True, 'dim': dim}
    generator_training = dp0.DataGenerator_z0(training_particle_IDs, training_labels_particle_IDS, s.sims_dic,
                                              shuffle=True, batch_size=64, **params_box)
    generator_validation = dp0.DataGenerator_z0(val_particle_IDs, val_labels_particle_IDS, s.sims_dic,
                                                gridded_box=generator_training.box_class, shuffle=False, batch_size=50,
                                                **params_box)