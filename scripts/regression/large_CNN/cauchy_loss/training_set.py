import sys
sys.path.append("/home/luisals/DeepHalos")
import dlhalos_code.data_processing as tn
from pickle import dump

if __name__ == "__main__":

    path_data = "/lfstev/deepskies/luisals/regression/large_CNN/cauchy_loss/l2/"

    # First you will have to load the simulation

    all_sims = ["0", "1", "2", "4", "5", "6"]
    s = tn.SimulationPreparation(all_sims)

    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    training_set = tn.InputsPreparation(train_sims, shuffle=True, scaler_type="minmax", load_ids=False,
                                        random_subset_each_sim=None, random_style="uniform",
                                        num_per_mass_bin=10000, num_bins=50)

    dump(training_set.particle_IDs, open(path_data + 'training_set.pkl', 'wb'))
    dump(training_set.labels_particle_IDS, open(path_data + 'labels_training_set.pkl', 'wb'))
    dump(training_set.scaler_output, open(path_data + 'scaler_output.pkl', 'wb'))

    v_set = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False, shuffle=True,
                                 random_style="random", random_subset_all=5000, random_subset_each_sim=None,
                                 scaler_output=training_set.scaler_output)
    dump(v_set.particle_IDs, open(path_data + 'validation_set.pkl', 'wb'))
    dump(v_set.labels_particle_IDS, open(path_data + 'labels_validation_set.pkl', 'wb'))

    v_set2 = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False,
                                 random_style="random", random_subset_all=50000, random_subset_each_sim=None,
                                 scaler_output=training_set.scaler_output)
    dump(v_set2.particle_IDs, open(path_data + 'larger_validation_set.pkl', 'wb'))
    dump(v_set2.labels_particle_IDS, open(path_data + 'larger_labels_validation_set.pkl', 'wb'))


