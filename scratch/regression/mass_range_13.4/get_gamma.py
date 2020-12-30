import numpy as np
import h5py
import os


path = "/mnt/beegfs/work/ati/pearl037/regression/mass_range_13.4/random_20sims_200K/"

dirs = ["alpha-2/", "alpha-2.5/", "alpha-3/"]

for dir in dirs:
    path_dir = path + dir
    epochs = ["%02d" % elem for elem in np.arange(1, len(os.listdir(path_dir + "model/.")) + 1)]
    g = [0.2]
    for num_epoch in epochs:
        w = path_dir + "model/weights." + num_epoch + ".h5"
        with h5py.File(w, "r") as f:
            g.append(list(f['loss_trainable_params']['loss_trainable_params']['gamma:0'])[0])
    np.save(path_dir + "gamma.npy", g)

