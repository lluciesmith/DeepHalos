import dlhalos_code.data_processing as tn
from pickle import dump
import numpy as np

if __name__ == "__main__":
    #path = "/mnt/beegfs/work/ati/pearl037/regression/bigbox/data/"
    path = "/freya/ptmp/mpa/luisals/deep_halos/"

    #path_sims = "/mnt/beegfs/work/ati/pearl037/"
    path_sims = "/freya/ptmp/mpa/luisals/Simulations/"
    all_sims = ["L200_N1024_genetIC", "L200_N1024_genetIC2", "L200_N1024_genetIC3"]
    train_sims = all_sims[:-1]
    val_sim = all_sims[-1]

    n_samples = 200000
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