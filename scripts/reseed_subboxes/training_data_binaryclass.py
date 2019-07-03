import numpy as np
import sys; sys.path.append("/home/lls/DeepHalos/")
import data_processing as dp


paths = ["/share/data1/lls/standard_reseed3/", "/share/data1/lls/standard_reseed4/", "/share/data1/lls/standard_reseed5"]

for path in paths:
    halo_mass = np.load(path + "halo_mass_particles.npy")
    particle_ids = np.array([line.rstrip("\n") for line in open(path + "subboxes_ids.txt")]).astype("int")

    labels = dp.transform_halo_mass_to_binary_classification(halo_mass, threshold=2*10**12)

    t_0 = np.concatenate((particle_ids[np.random.choice(np.where(labels[particle_ids] == 0)[0], 10000, replace=False)],
                          particle_ids[np.random.choice(np.where(labels[particle_ids] == 1)[0], 10000, replace=False)]))
    # np.random.shuffle(t_0)
    np.savetxt(path + "training_ids_binary_classification.txt", t_0, fmt="%i", delimiter=",")
