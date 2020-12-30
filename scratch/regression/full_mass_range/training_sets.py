import dlhalos_code.data_processing as tn
from pickle import dump

if __name__ == "__main__":
    path_sims = "/mnt/beegfs/work/ati/pearl037/"
    num_sims = 9

    if num_sims == 9:
        all_sims = ["0", "1", "2", "4", "5", "7", "8", "9", "10", "6"]
        path_random = "/mnt/beegfs/work/ati/pearl037/regression/training_set/9sims/random/"
        path_uniform = "/mnt/beegfs/work/ati/pearl037/regression/training_set/9sims/uniform/"
    else:
        all_sims = ["0", "1", "2", "4", "5", "6"]
        path_random = "/mnt/beegfs/work/ati/pearl037/regression/training_set/random/"
        path_uniform = "/mnt/beegfs/work/ati/pearl037/regression/training_set/uniform/"

    s = tn.SimulationPreparation(all_sims, path=path_sims)
    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    # Save training sets sampled at random from 5 simulations, with n=50,000 and n=500,000

    paths = [path_random + "200k/", path_random + "500k/"]
    samples = [200000, 500000]

    for i, n_samples in enumerate(samples):

        saving_path = paths[i]

        training_set = tn.InputsPreparation(train_sims, shuffle=True, scaler_type="minmax", return_rescaled_outputs=True,
                                            output_range=(-1, 1), load_ids=False, random_subset_each_sim=None,
                                            random_style="random", random_subset_all=n_samples, path=path_sims)
        dump(training_set.particle_IDs, open(saving_path + 'training_set.pkl', 'wb'))
        dump(training_set.labels_particle_IDS, open(saving_path + 'labels_training_set.pkl', 'wb'))
        dump(training_set.scaler_output, open(saving_path + 'scaler_output.pkl', 'wb'))

        v_set = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False, shuffle=True,
                                     random_style="random", random_subset_all=5000, random_subset_each_sim=1000000,
                                     scaler_output=training_set.scaler_output, path=path_sims)
        dump(v_set.particle_IDs, open(saving_path + 'validation_set.pkl', 'wb'))
        dump(v_set.labels_particle_IDS, open(saving_path + 'labels_validation_set.pkl', 'wb'))

        v_set2 = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False,
                                     random_style="random", random_subset_all=50000, random_subset_each_sim=1000000,
                                     scaler_output=training_set.scaler_output, path=path_sims)
        dump(v_set2.particle_IDs, open(saving_path + 'larger_validation_set.pkl', 'wb'))
        dump(v_set2.labels_particle_IDS, open(saving_path + 'larger_labels_validation_set.pkl', 'wb'))

    # Save training sets sampled uniformly from 50 bins, with n=1000 or n=10000 per mass bin

    paths = [path_uniform + "4k_perbin/", path_uniform + "10k_perbin/"]
    samples = [4000, 10000]

    for i, n_samples in enumerate(samples):
        saving_path = paths[i]

        training_set = tn.InputsPreparation(train_sims, shuffle=True, scaler_type="minmax", return_rescaled_outputs=True,
                                            output_range=(-1, 1), load_ids=False, random_subset_each_sim=None,
                                            random_style="uniform", num_per_mass_bin=n_samples, num_bins=50,
                                            path=path_sims)

        dump(training_set.particle_IDs, open(saving_path + 'training_set.pkl', 'wb'))
        dump(training_set.labels_particle_IDS, open(saving_path + 'labels_training_set.pkl', 'wb'))
        dump(training_set.scaler_output, open(saving_path + 'scaler_output.pkl', 'wb'))

        v_set = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False, shuffle=True,
                                     random_style="random", random_subset_all=5000, random_subset_each_sim=1000000,
                                     scaler_output=training_set.scaler_output, path=path_sims)
        dump(v_set.particle_IDs, open(saving_path + 'validation_set.pkl', 'wb'))
        dump(v_set.labels_particle_IDS, open(saving_path + 'labels_validation_set.pkl', 'wb'))

        v_set2 = tn.InputsPreparation([val_sim], scaler_type="minmax", load_ids=False,
                                     random_style="random", random_subset_all=50000, random_subset_each_sim=1000000,
                                     scaler_output=training_set.scaler_output, path=path_sims)
        dump(v_set2.particle_IDs, open(saving_path + 'larger_validation_set.pkl', 'wb'))
        dump(v_set2.labels_particle_IDS, open(saving_path + 'larger_labels_validation_set.pkl', 'wb'))


