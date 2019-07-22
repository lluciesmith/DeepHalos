import numpy as np
#
# training_ids = np.load("/share/data2/lls/deep_halos/training_ids.npy")
# t = list(training_ids.astype("str"))

import re
import os

val_ids = []
regex = re.compile(r'\d+')
for filename in os.listdir("subboxes/")[:-1]:
    val_ids.append(int(regex.findall(filename)[-1]))

val_ids = np.array(val_ids)
print(len(val_ids))
np.save("reseed5_subboxes_ids.npy", val_ids)

