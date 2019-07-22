import numpy as np
import data_processing as dp


def get_ids_and_binary_class_labels(sim="0", threshold=2*10**12, ids_filename="training_ids_binary_classification.txt",
                                    path="/content/drive/My Drive/"):
    if sim == "0":
        path1 = path + "new_training_sim/"
        halo_mass = np.load(path + "training_simulation/halo_mass_particles.npy")
    else:
        path1 = path + "reseed" + sim + "_simulation/reseed"+ sim + "_"
        halo_mass = np.load(path + "reseed" + sim + "_simulation/reseed" + sim + "_halo_mass_particles.npy")

    labels_all = dp.transform_halo_mass_to_binary_classification(halo_mass, threshold=threshold)

    with open(path1 + ids_filename, "r") as f:
      ids_bc = np.array([line.rstrip("\n") for line in f]).astype("int")

    np.random.shuffle(ids_bc)
    return ids_bc, labels_all[ids_bc]


def create_generator_sim(ids_sim, labels_sim, path="training", batch_size=40):
    partition = {'ids': list(ids_sim.astype("str"))}
    labels_dic = dict(zip(list(ids_sim.astype("str")), labels_sim))

    gen_params = {'dim': (51, 51, 51), 'batch_size': batch_size, 'n_channels': 1, 'shuffle': False}

    training_generator = dp.DataGenerator(partition['ids'], labels_dic, **gen_params,
                                          saving_path=path, model_type="binary_classification")
    return training_generator


# create a list of names for particles ID in each simulation

def create_generator_multiple_sims(list_sims, list_ids_per_sim, labels_sim, batch_size=40):
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
                                          **gen_params, multiple_sims=True,
                                          model_type="binary_classification")
    return training_generator



