import dlhalos_code.data_processing as tn
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

    shell_labels = tn.assign_shell_to_pixels(res, params.params_box['num_shells'])

    saving_path = "/share/hypatia/lls/newdlhalos/training_data/"
    training_ids = load(open(saving_path + 'training_set.pkl', 'rb'))
    validation_ids = load(open(saving_path + 'validation_set.pkl', 'rb'))
    testing_ids = load(open(saving_path + 'test_set.pkl', 'rb'))
    for pids in [training_ids, validation_ids, testing_ids]:
        for ID in pids:
            sim_index = ID[ID.find('sim-') + len('sim-'): ID.find('-id')]
            particle_ID = ID[ID.find('-id-') + + len('-id-'):]
            if os.path.exists(saving_path + "/inputs_raw/inp_raw_sim_" + sim_index + "_particle_" + particle_ID + ".npy"):
                pass
            else:
            
                i0, j0, k0 = s.sims_dic[sim_index]['coords'][int(particle_ID)]
                delta_sim = sims_rescaled_density[sim_index]
                output_matrix = np.zeros((res, res, res))
                box = tn.compute_subbox(i0, j0, k0, res, delta_sim, output_matrix, shape_sim)
                np.save(saving_path + "/inputs_raw/inp_raw_sim_" + sim_index + "_particle_" + particle_ID + ".npy", box)
                if params.params_box['input_type'] == "averaged":
                    box = tn.get_spherically_averaged_box(box, shell_labels)
                    np.save(saving_path + "/inputs_avg/inp_avg_sim_" + sim_index + "_particle_" + particle_ID + ".npy", box)