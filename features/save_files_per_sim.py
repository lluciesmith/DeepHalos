import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
from utils import generators_training as gbc


def create_one_single_array_features(sim, dim=(75, 75, 75), ph="/lfstev/deepskies/luisals/"):
    f = "random_training_set.txt"

    if sim == "0":
        path = ph + "training_simulation/training_set_res75/"

    else:
        path = ph + "reseed" + sim + "_simulation/training_set_res75/"

    ids_i, mass_i = gbc.get_ids_and_regression_labels(sim=sim, ids_filename=f, fitted_scaler=None, shuffle=False)
    generator_i = gbc.create_generator_sim(ids_i, mass_i, batch_size=len(ids_i), dim=dim, z=99,
                                           path=path)
    Xi, yi = generator_i[0]
    # df = pd.Dataframe(X0, columns=generator_0.list_IDs)
    np.save(path + "res_75_20000_subboxes.npy", Xi)
    assert np.allclose(yi, mass_i)


if __name__ == "__main__":
    sims = ["0", "1", "2", "4", "5", "6"]

    for sim_i in sims:
        print("Doing simulation " + sim_i)
        create_one_single_array_features(sim_i, dim=(75, 75, 75), ph="/lfstev/deepskies/luisals/")
