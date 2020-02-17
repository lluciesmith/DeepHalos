import numpy as np
import sys; sys.path.append("/home/lls/mlhalos_code/")
sys.path.append("/home/lls/")
# import sys; sys.path.append("/Users/lls/Documents/mlhalos_code/")
# sys.path.append("/Users/lls/Documents/Projects/")
from mlhalos import parameters
from DeepHalos import subboxes as subb
from multiprocessing import Pool
import gc
import os


def halo_mass_particles(initial_parameters):
    f = initial_parameters.final_snapshot
    halo_ids = f["grp"]
    h_mass = np.zeros(len(halo_ids),)

    for i in range(len(initial_parameters.halo)):
        ind = np.where(halo_ids == i)[0]
        h_mass[ind] = initial_parameters.halo[i]['mass'].sum()

    return h_mass


def compute_and_save_subbox_particle(particle_id):
    try:
        delta_sub = sub_in.get_qty_in_subbox(particle_id)
    except:
        print("This failed for particle " + str(particle_id) + "so doing proper SPH")
        delta_sub = sub_in.get_sph_particle(particle_id)

    os.makedirs(saving_path + str(particle_id))
    np.save(saving_path + "subboxes/" + str(particle_id) +
            "/subbox_" + str(res) +"_particle_" + str(particle_id) + ".npy", delta_sub)
    del delta_sub
    gc.collect()


if __name__ == "__main__":
    sims = ["6", "7", "8", "9", "10"]
    for i in range(5):
        sim = sims[i]

        path_sim = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
        saving_path = "/share/hypatia/lls/deep_halos/reseed_" + sim + "/"

        initial_params = parameters.InitialConditionsParameters(initial_snapshot=path_sim + "IC.gadget2",
                                                                final_snapshot=path_sim + "output/snapshot_007",
                                                                load_final=True)

        # Save halo mass of particles and select training samples
        particle_ids = initial_params.final_snapshot["iord"]

        try:
            print("Load halo mass particles")
            halo_mass_ids = np.load(saving_path + "halo_mass_particles.npy")
        except:
            print("Computing halo mass particles")
            halo_mass_ids = halo_mass_particles(initial_params)
            np.save(saving_path + "halo_mass_particles.npy", halo_mass_ids)

        particle_ids_in_halos = particle_ids[halo_mass_ids > 0]
        halo_mass_p_ids_in_halos = halo_mass_ids[halo_mass_ids > 0]

        try:
            print("Loading training set")
            training_set = np.loadtxt(saving_path + "reseed_" + sim + "_random_training_set.txt", dtype="int",
                                      delimiter=",")
        except:
            print("Computing training set")
            training_set = np.random.choice(particle_ids_in_halos, 20000, replace=False)
            np.savetxt(saving_path + "reseed_" + sim + "_random_training_set.txt", training_set, fmt="%i", delimiter=",")

        # Compute subboxes for particles

        res = 51
        sub_in = subb.Subboxes(initial_params, subbox_shape=(res, res, res))

        pool = Pool(processes=80)
        pool.map(compute_and_save_subbox_particle, training_set)
        pool.close()

