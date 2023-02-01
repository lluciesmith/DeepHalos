import dlhalos_code.data_processing as tn
from pickle import dump
import numpy as np

if __name__ == "__main__":
    path_sims = "/share/hypatia/lls/simulations/dlhalos_sims/"
    all_sims = ["%i" % i for i in np.arange(25)]
    all_sims.remove("3")
    val_sim = ["6"]
    test_sim = ["6", "22", "23", "24"]
    train_sims = list(np.array(all_sims)[np.where(~np.in1d(all_sims, test_sim) & ~np.in1d(all_sims, val_sim))[0]])

    # training set
    n_samples = 200000
    saving_path = "/share/hypatia/lls/newdlhalos/training_data/"

    training_set = tn.InputsPreparation(train_sims, shuffle=True, scaler_type="minmax", return_rescaled_outputs=True,
                                        output_range=(-1, 1), log_high_mass_limit=13.4,
                                        load_ids=False, random_subset_each_sim=None,
                                        random_style="random", random_subset_all=n_samples,
                                        path=path_sims)
    dump(training_set.particle_IDs, open(saving_path + 'training_set.pkl', 'wb'))
    dump(training_set.labels_particle_IDS, open(saving_path + 'labels_training_set.pkl', 'wb'))
    dump(training_set.scaler_output, open(saving_path + 'scaler_output.pkl', 'wb'))

    # validation set
    v_set = tn.InputsPreparation(val_sim, scaler_type="minmax", load_ids=False, shuffle=True,
                                 random_style="random", log_high_mass_limit=13.4,
                                 random_subset_all=5000, random_subset_each_sim=None,
                                 scaler_output=training_set.scaler_output, path=path_sims)
    dump(v_set.particle_IDs, open(saving_path + 'validation_set.pkl', 'wb'))
    dump(v_set.labels_particle_IDS, open(saving_path + 'labels_validation_set.pkl', 'wb'))

    # test set
    test_set = tn.InputsPreparation(test_sim, scaler_type="minmax", load_ids=False, log_high_mass_limit=13.4,
                                    random_style="random", random_subset_all=50000, random_subset_each_sim=None,
                                    scaler_output=training_set.scaler_output, path=path_sims)
    # see if there are test set particles from sim 6 that are also in the validation set
    idx = ~np.in1d(test_set.particle_IDs, v_set.particle_IDs)
    test_setIDs = list(np.array(test_set.particle_IDs)[np.where(idx)[0]])
    labels_test_setIDs = {key: test_set.labels_particle_IDS[key] for key in test_setIDs}
    dump(test_setIDs, open(saving_path + 'test_set.pkl', 'wb'))
    dump(labels_test_setIDs, open(saving_path + 'labels_test_set.pkl', 'wb'))
