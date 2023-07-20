import numpy as np
import sys
sys.path.append("/home/lls/mlhalos_code/")
from mlhalos import parameters, window
sys.path.append("/home/lls/mlhalos_code/scripts/")
#from ellipsoidal import ellipsoidal_barrier as eb
import analytic_barriers as ab


def EPS_mass_prediction(trajectories, smoothing_mass, initial_parameters):
    delta_sc = ab.get_spherical_collapse_barrier(initial_parameters.initial_conditions, z=99, delta_sc_0=1.686, output="rho/rho_bar")
    #delta_sc = ab.get_spherical_collapse_barrier(initial_parameters, z=99, delta_sc_0=1.686, output="rho/rho_bar",
    #                                             growth=None)
    EPS_mass = np.array([smoothing_mass[(np.where(traj >= delta_sc)[0]).max()]
                         if any([num >= delta_sc for num in traj]) else -1 for traj in trajectories])
    return EPS_mass


def ST_mass_prediction(trajectories, smoothing_mass, initial_parameters, beta=0.485, gamma=0.615, a=0.707):
    ellip_threshold = ab.ellipsoidal_collapse_barrier(smoothing_mass, initial_parameters.initial_conditions,
                                                      z=99, beta=beta, gamma=gamma, a=a, output="rho/rho_bar")
    # ellip_threshold = ab.ellipsoidal_collapse_barrier(smoothing_mass, initial_parameters,
    #                                                   beta=beta, gamma=gamma, a=a, z=99)
    ST_mass = np.array([smoothing_mass[(np.where(trajectories[i] >= ellip_threshold)[0]).max()]
                         if any(trajectories[i] >= ellip_threshold) else -1 for i in range(len(trajectories))])
    return ST_mass


if __name__ == "__main__":
    for sim in ["22", "23", "24"]:
        saving_path = "/share/data1/lls/standard_reseed" + sim + "/"

        traj = np.load(saving_path + "density_contrasts.npy")
        path_simulation5 = "/share/hypatia/lls/simulations/standard_reseed" + sim + "/"
        ic = parameters.InitialConditionsParameters(initial_snapshot=path_simulation5 + "IC.gadget2",
                                                    load_final=False, min_halo_number=0, max_halo_number=400,
                                                    min_mass_scale=3e10, max_mass_scale=1e15)
        # Get features for each particles
        w = window.WindowParameters(initial_parameters=ic, num_filtering_scales=50, snapshot=None, volume="sphere")
        m = w.smoothing_masses

        ps_mass_testing = EPS_mass_prediction(traj, m, ic)
        np.save(saving_path + "EPS_predictions_sim" + sim + ".npy", ps_mass_testing)
        st_mass_testing = ST_mass_prediction(traj, m, ic)
        np.save(saving_path + "ST_predictions_sim" + sim + ".npy", st_mass_testing)