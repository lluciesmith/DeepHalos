import dlhalos_code.data_processing as tn
from pickle import dump
import numpy as np

if __name__ == "__main__":
    path_random = "/mnt/beegfs/work/ati/pearl037/regression/training_set_13.4/20sims/random/"
    path_uniform = "/mnt/beegfs/work/ati/pearl037/regression/training_set_13.4/20sims/uniform/"

    path_sims = "/mnt/beegfs/work/ati/pearl037/"
    all_sims = ["%i" % i for i in np.arange(22)]
    all_sims.remove("3")
    all_sims.remove("6")
    all_sims.append("6")
    s = tn.SimulationPreparation(all_sims, path=path_sims)
    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    # Save training sets sampled at random from 9 simulations, with n=200,000

    n_samples = 200000
    saving_path = path_random + "200k/"

    training_set = tn.InputsPreparation(train_sims, shuffle=True, scaler_type="minmax", return_rescaled_outputs=True,
                                        output_range=(-1, 1), log_high_mass_limit=13.4,
                                        load_ids=False, random_subset_each_sim=None,
                                        random_style="random", random_subset_all=n_samples,
                                        path=path_sims)

    dump(training_set.particle_IDs, open(saving_path + 'training_set.pkl', 'wb'))
    dump(training_set.labels_particle_IDS, open(saving_path + 'labels_training_set.pkl', 'wb'))
    dump(training_set.scaler_output, open(saving_path + 'scaler_output.pkl', 'wb'))

    v_set = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False, shuffle=True,
                                 random_style="random", log_high_mass_limit=13.4,
                                 random_subset_all=5000, random_subset_each_sim=1000000,
                                 scaler_output=training_set.scaler_output, path=path_sims)
    dump(v_set.particle_IDs, open(saving_path + 'validation_set.pkl', 'wb'))
    dump(v_set.labels_particle_IDS, open(saving_path + 'labels_validation_set.pkl', 'wb'))

    v_set2 = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False, log_high_mass_limit=13.4,
                                 random_style="random", random_subset_all=50000, random_subset_each_sim=1000000,
                                 scaler_output=training_set.scaler_output, path=path_sims)
    dump(v_set2.particle_IDs, open(saving_path + 'larger_validation_set.pkl', 'wb'))
    dump(v_set2.labels_particle_IDS, open(saving_path + 'larger_labels_validation_set.pkl', 'wb'))

    del saving_path
    del n_samples
    del training_set
    del v_set
    del v_set2

    # Save training sets sampled uniformly from 80 bins, with n=4000 per mass bin

    saving_path = path_uniform + "2.5k_in_each_80bins/"
    n_samples = 2500

    training_set = tn.InputsPreparation(train_sims, shuffle=True, scaler_type="minmax", return_rescaled_outputs=True,
                                        output_range=(-1, 1), load_ids=False, random_subset_each_sim=None,
                                        log_high_mass_limit=13.4,
                                        random_style="uniform", num_per_mass_bin=n_samples, num_bins=80,
                                        path=path_sims)

    dump(training_set.particle_IDs, open(saving_path + 'training_set.pkl', 'wb'))
    dump(training_set.labels_particle_IDS, open(saving_path + 'labels_training_set.pkl', 'wb'))
    dump(training_set.scaler_output, open(saving_path + 'scaler_output.pkl', 'wb'))

    v_set = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False, shuffle=True,
                                 random_style="random", random_subset_all=5000, random_subset_each_sim=1000000,
                                 log_high_mass_limit=13.4,
                                 scaler_output=training_set.scaler_output, path=path_sims)
    dump(v_set.particle_IDs, open(saving_path + 'validation_set.pkl', 'wb'))
    dump(v_set.labels_particle_IDS, open(saving_path + 'labels_validation_set.pkl', 'wb'))

    v_set2 = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False,
                                  log_high_mass_limit=13.4, random_style="random", random_subset_all=50000,
                                  random_subset_each_sim=1000000,
                                  scaler_output=training_set.scaler_output, path=path_sims)
    dump(v_set2.particle_IDs, open(saving_path + 'larger_validation_set.pkl', 'wb'))
    dump(v_set2.labels_particle_IDS, open(saving_path + 'larger_labels_validation_set.pkl', 'wb'))


