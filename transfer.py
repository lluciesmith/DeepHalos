import numpy as np
#
# training_ids = np.load("/share/data2/lls/deep_halos/training_ids.npy")
# t = list(training_ids.astype("str"))

import re
import os

path = "/share/hypatia/lls/deep_halos/"
dirs = [path + "training_sim/training_set/",
        path + "reseed_1/training_set/",
        path + "reseed_2/training_set/",
        path + "reseed_3/training_set/",
        path + "reseed_4/training_set/",
        path + "reseed_5/training_set/"]

for i in range(len(dirs)):
    dir = dirs[i]
    val_ids = []
    regex = re.compile(r'\d+')
    for filename in os.listdir(dir)[:-1]:
        val_ids.append(int(regex.findall(filename)[-1]))

    val_ids = np.array(val_ids)
    print(len(val_ids))
    np.save(dir + "../saved_ids_training_set.txt", val_ids)
