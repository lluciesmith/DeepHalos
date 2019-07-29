import numpy as np
import data_processing as dp


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


def get_ids_and_regression_labels(sim="0", ids_filename="balanced_training_set.txt",
                                  path="/lfstev/deepskies/luisals/", fitted_scaler=None):
    if sim == "0":
        path1 = path + "training_simulation/training_sim_"
        halo_mass = np.load(path + "training_simulation/halo_mass_particles.npy")
    else:
        path1 = path + "reseed" + sim + "_simulation/reseed_" + sim + "_"
        halo_mass = np.load(path + "reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")

    with open(path1 + ids_filename, "r") as f:
      ids_bc = np.array([line.rstrip("\n") for line in f]).astype("int")

    np.random.shuffle(ids_bc)

    if fitted_scaler is None:
        scaler_output, output_all = dp.normalise_output(halo_mass, take_log=True)
        return ids_bc, output_all[ids_bc], scaler_output

    else:
        output_all = np.zeros((len(halo_mass),))
        log_mass = np.log10(halo_mass[halo_mass > 0])
        output_all[halo_mass > 0] = fitted_scaler.transform(log_mass.reshape(-1, 1)).flatten()
        return ids_bc, output_all[ids_bc]


def create_generator_sim(ids_sim, labels_sim, path="training", batch_size=40):
    partition = {'ids': list(ids_sim.astype("str"))}
    labels_dic = dict(zip(list(ids_sim.astype("str")), labels_sim))

    gen_params = {'dim': (51, 51, 51), 'batch_size': batch_size, 'n_channels': 1, 'shuffle': False}

    training_generator = dp.DataGenerator(partition['ids'], labels_dic, **gen_params,
                                          saving_path=path)
    return training_generator


# create a list of names for particles ID in each simulation

def create_generator_multiple_sims(list_sims, list_ids_per_sim, labels_sim, batch_size=40,
                                   path="/lfstev/deepskies/luisals/"):
    partition = {'ids': []}
    labels_dic = {}
    for i in range(len(list_sims)):
        sim_i = list_sims[i]
        ids_sim_i = list_ids_per_sim[i]
        labels_sim_i = labels_sim[i]

        name = ['sim-' + str(sim_i) + '-id-' + str(id_i) for id_i in ids_sim_i]
        for val in name:
            partition['ids'].append(val)

        dict_i = dict(zip(name, labels_sim_i))
        labels_dic.update(dict_i)

    gen_params = {'dim': (51, 51, 51), 'batch_size': batch_size, 'n_channels': 1,
                  'shuffle': False}

    training_generator = dp.DataGenerator(partition['ids'], labels_dic,
                                          **gen_params, multiple_sims=True, saving_path=path)
    return training_generator



