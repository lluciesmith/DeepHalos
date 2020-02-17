import numpy as np
import data_processing as dp
import sklearn.preprocessing
import time


def get_standard_scaler_and_transform(list_outputs):
    outputs_conc = np.concatenate(list_outputs)
    norm_scaler = sklearn.preprocessing.StandardScaler()
    norm_scaler.fit(outputs_conc.reshape(-1, 1))
    rescaled_out = [transform_array_given_scaler(norm_scaler, out_i) for out_i in list_outputs]
    return rescaled_out, norm_scaler


def transform_array_given_scaler(scaler, array):
    scaled_array = scaler.transform(array.reshape(-1, 1)).flatten()
    return scaled_array


def get_ids_and_regression_labels(sim="0", ids_filename="balanced_training_set.txt",
                                  path="/lfstev/deepskies/luisals/", fitted_scaler=None, shuffle=True):
    if sim == "0":
        path1 = path + "training_simulation/training_sim_"
        halo_mass = np.load(path + "training_simulation/halo_mass_particles.npy")
    else:
        path1 = path + "reseed" + sim + "_simulation/reseed_" + sim + "_"
        halo_mass = np.load(path + "reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")

    with open(path1 + ids_filename, "r") as f:
      ids_bc = np.array([line.rstrip("\n") for line in f]).astype("int")

    if shuffle is True:
        np.random.seed(4)
        np.random.shuffle(ids_bc)

    if fitted_scaler is not None:
        output_ids = transform_array_given_scaler(fitted_scaler, np.log10(halo_mass[ids_bc]))
    else:
        output_ids = np.log10(halo_mass[ids_bc])

    return ids_bc, output_ids


################# GENERATORS ##################


def create_generator_sim(ids_sim, labels_sim, path="training", batch_size=40, rescale_mean=0, rescale_std=1, z=99,
                         dim=(51, 51, 51)):
    partition = {'ids': list(ids_sim.astype("str"))}
    labels_dic = dict(zip(list(ids_sim.astype("str")), labels_sim))

    gen_params = {'dim': dim, 'batch_size': batch_size, 'n_channels': 1, 'shuffle': False}

    training_generator = dp.DataGenerator(partition['ids'], labels_dic, **gen_params,
                                          saving_path=path, rescale_mean=rescale_mean, rescale_std=rescale_std, z=z)
    return training_generator


# create a list of names for particles ID in each simulation

def create_generator_multiple_sims(list_sims, list_ids_per_sim, labels_sim, batch_size=40, dim=(51, 51, 51),
                                   path="/lfstev/deepskies/luisals/", rescale_mean=0, rescale_std=1, z=99):
    t0 = time.time()

    labels_dic = {}
    for i in range(len(list_sims)):
        sim_i = list_sims[i]
        ids_sim_i = list_ids_per_sim[i]
        labels_sim_i = labels_sim[i]

        name = ['sim-' + str(sim_i) + '-id-' + str(id_i) for id_i in ids_sim_i]
        dict_i = dict(zip(name, labels_sim_i))
        labels_dic.update(dict_i)

    np.random.seed(5)
    ids_reordering = np.random.permutation(list(labels_dic.keys()))
    labels_reordered = dict([(key, labels_dic[key]) for key in ids_reordering])
    partition = {'ids': list(labels_reordered.keys())}

    gen_params = {'dim': dim, 'batch_size': batch_size, 'n_channels': 1,
                  'shuffle': False}

    t1 = time.time()
    print("Setting up parameters for data generator took " + str((t1 - t0) / 60) + " minutes.")

    training_generator = dp.DataGenerator(partition['ids'], labels_reordered, **gen_params,
                                          rescale_mean=rescale_mean, rescale_std=rescale_std,
                                          multiple_sims=True, saving_path=path, z=z)
    return training_generator


# def compute_mean_std_inputs(list_sims, list_ids_per_sim, labels_sim, batch_size=40,
#                                    path="/lfstev/deepskies/luisals/", z=99):
#     gen = create_generator_multiple_sims(list_sims, list_ids_per_sim, labels_sim, batch_size=batch_size,
#                                    path=path, rescale_mean=0, rescale_std=1, z=z)
#     num_batches = len(gen.indexes)/batch_size
#
#     means = []
#     for i in range(num_batches):
#         # takes forever....
#
#         bi = gen[i]
#         means.append(np.mean(bi[0]))
#         del bi
#
#     mean_inputs = np.mean(means)
#     std = np.std()
#
#     return np.mean(means)


# Spherical overdensities functions


def get_ids_and_SO_labels(sim="0", ids_filename="balanced_training_set.txt", path="/lfstev/deepskies/luisals/",
                          fitted_scaler=None, index=34):
    if sim == "0":
        path1 = path + "training_simulation/training_sim_"
        tr_i = np.load(path + "training_simulation/traj_index_" + str(index) + ".npy")
    else:
        path1 = path + "reseed" + sim + "_simulation/reseed_" + sim + "_"
        tr_i = np.load(path + "reseed" + sim + "_simulation/traj_index_" + str(index) + ".npy")

    with open(path1 + ids_filename, "r") as f:
      ids_bc = np.array([line.rstrip("\n") for line in f]).astype("int")

    np.random.seed(4)
    np.random.shuffle(ids_bc)

    if fitted_scaler is not None:
        output_ids = transform_array_given_scaler(fitted_scaler, tr_i[ids_bc])
    else:
        output_ids = tr_i[ids_bc]

    return ids_bc, output_ids


# Binary classification


def get_ids_and_binary_class_labels(sim="0", threshold=2*10**12, ids_filename="training_ids_binary_classification.txt",
                                    path="/lfstev/deepskies/luisals/"):
    if sim == "0":
        path1 = path + "training_simulation/"
        halo_mass = np.load(path + "training_simulation/halo_mass_particles.npy")
    else:
        path1 = path + "reseed" + sim + "_simulation/reseed" + sim + "_"
        halo_mass = np.load(path + "reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")

    labels_all = dp.transform_halo_mass_to_binary_classification(halo_mass, threshold=threshold)

    with open(path1 + ids_filename, "r") as f:
      ids_bc = np.array([line.rstrip("\n") for line in f]).astype("int")

    np.random.shuffle(ids_bc)
    return ids_bc, labels_all[ids_bc]



