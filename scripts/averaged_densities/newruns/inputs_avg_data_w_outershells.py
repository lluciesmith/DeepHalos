import dlhalos_code_tf2.data_processing as tn
from pickle import load
import numpy as np
import os

if __name__ == "__main__":
    import params_avg as params

    path_sims = "/share/hypatia/lls/simulations/dlhalos_sims/"
    all_sims = ["%i" % i for i in np.arange(25)]
    all_sims.remove("3")
    s = tn.SimulationPreparation(all_sims, path=path_sims)
    shape_sim = int(round((s.sims_dic["6"]["iord"].shape[0]) ** (1 / 3)))
    res = params.dim[0]

    sims_rescaled_density = {}
    for i, simulation in s.sims_dic.items():
        d = (simulation["den_contrast"] - params.params_tr['rescale_mean']) / params.params_tr['rescale_std']
        sims_rescaled_density[i] = d.reshape((shape_sim, shape_sim, shape_sim))

    deltaR = np.diff(np.linspace(2, res / 2, params.params_box['num_shells'], endpoint=True))[0]
    r_shells2 = np.arange(2, tn.get_r_coords(res).max() + 10, deltaR)
    shell_labels2 = tn.assign_shell_to_pixels(res, params.params_box['num_shells'], r_shells=r_shells2)

    saving_path = "/share/hypatia/lls/newdlhalos/training_data/"
    training_ids = load(open(saving_path + 'training_set_400k.pkl', 'rb'))
    validation_ids = load(open(saving_path + 'validation_set_400k.pkl', 'rb'))
    testing_ids = load(open(saving_path + 'test_set_400k.pkl', 'rb'))
    for pids in [training_ids, validation_ids, testing_ids]:
        for ID in pids:
            sim_index = ID[ID.find('sim-') + len('sim-'): ID.find('-id')]
            particle_ID = ID[ID.find('-id-') + + len('-id-'):]
            partfilename = saving_path + "/inputs_avg_wouter/inp_avg_sim_" + sim_index + "_particle_" + particle_ID + ".npy"
            if os.path.exists(partfilename):
                pass
            else:
                print(ID)
                box = np.load(saving_path + "/inputs_raw/inp_raw_sim_" + sim_index + "_particle_" + particle_ID + ".npy")
                if params.params_box['input_type'] == "averaged":
                    box = tn.get_spherically_averaged_box(box, shell_labels2)
                    np.save(partfilename, box)


    # saving_path = "/share/hypatia/lls/newdlhalos/training_data/"
    # training_ids = load(open(saving_path + 'training_set.pkl', 'rb'))
    # validation_ids = load(open(saving_path + 'validation_set.pkl', 'rb'))
    # testing_ids = load(open(saving_path + 'test_set.pkl', 'rb'))
    # for pids in [training_ids, validation_ids, testing_ids]:
    #     for ID in pids:
    #         sim_index = ID[ID.find('sim-') + len('sim-'): ID.find('-id')]
    #         particle_ID = ID[ID.find('-id-') + + len('-id-'):]
    #         if not os.path.exists(saving_path + "/inputs_avg/inp_avg_sim_" + sim_index + "_particle_" + particle_ID + ".npy"):
    #             print(ID)
