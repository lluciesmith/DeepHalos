import numpy as np

sims = ["0", "1", "2", "3", "4", "5"]
ids_saved_sims = ["/share/hypatia/lls/deep_halos/training_sim/saved_subboxes_ids.npy",
                  "/share/hypatia/lls/deep_halos/reseed_1/reseed1_subboxes_ids.npy",
                  "/share/hypatia/lls/deep_halos/reseed_2/reseed2_subboxes_ids.npy",
                  "/share/hypatia/lls/deep_halos/reseed_3/reseed3_subboxes_ids.npy",
                  "/share/hypatia/lls/deep_halos/reseed_4/reseed4_subboxes_ids.npy",
                  "/share/hypatia/lls/deep_halos/reseed_5/reseed5_subboxes_ids.npy"]

halo_masses_sims = ["/home/lls/stored_files/halo_mass_particles.npy",
                    "/share/data1/lls/reseed50/halo_mass_particles.npy",
                    "/share/data1/lls/reseed50_2/halo_mass_particles.npy",
                    "/share/data1/lls/standard_reseed3/halo_mass_particles.npy",
                    "/share/data1/lls/standard_reseed4/halo_mass_particles.npy",
                    "/share/data1/lls/standard_reseed5/halo_mass_particles.npy"]

paths = []

for i, sim in enumerate(sims):
    halo_mass = np.load(halo_masses_sims[i])
    ids_saved = np.load(ids_saved_sims[i])

    ids_in_h = np.where(halo_mass > 0)[0]

    ran_ids = np.random.choice(ids_in_h, 20000, replace=False)
    if sim == "0":
        np.savetxt("/share/hypatia/lls/deep_halos/training_sim/training_sim_random_training_set_regression.txt",
                   ran_ids, fmt="%i", delimiter=",")
    else:
        np.savetxt("/share/hypatia/lls/deep_halos/reseed_" + sim + "/reseed_" + sim +
                   "_random_training_set_regression.txt", ran_ids, fmt="%i", delimiter=",")


    # log_mass = np.log10(halo_mass[ids_in_h])
    # b = np.linspace(log_mass.min(), log_mass.max(), 50)
    #
    # ids_training = []
    # for i in range(49):
    #     ids_i = ids_saved[(np.log10(halo_mass[ids_saved]) >= b[i]) & (np.log10(halo_mass[ids_saved]) < b[i+1])]
    #     if ids_i.size:
    #         if len(ids_i) < 500:
    #             ids_training.append(ids_i)
    #         else:
    #             print("Bin" + str(i))
    #             ids_training.append(np.random.choice(ids_i, 500, replace=False))
    #
    # ids_training = np.concatenate(ids_training)
    # np.random.shuffle(ids_training)
    # print(len(ids_training))
    # if sim == "0":
    #     np.savetxt("/share/hypatia/lls/deep_halos/training_sim/training_sim_balanced_training_set.txt", ids_training,
    #                fmt="%i", delimiter=",")
    # else:
    #     np.savetxt("/share/hypatia/lls/deep_halos/reseed_" + sim + "/reseed_" + sim + "_balanced_training_set.txt",
    #                ids_training, fmt="%i", delimiter=",")

