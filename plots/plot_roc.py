import sys
sys.path.append("/home/luisals/DeepHalos")
import numpy as np
import CNN
from tensorflow import set_random_seed
from utils import generators_training as gbc
from tensorflow.keras.models import load_model


ph = "/lfstev/deepskies/luisals/"

ids_1, labels_1 = gbc.get_ids_and_binary_class_labels(sim="1", threshold=2 * 10 ** 12)
generator_1 = gbc.create_generator_sim(ids_1, labels_1, batch_size=80,
                                       path=ph + "reseed1_simulation/reseed1_training/")

model_4 = load_model("/lfstev/deepskies/luisals/binary_classification/train_sequential_sim02345/models/"
                     "model_177_epochs_train_sims0345_3epochs_per_sim.h5")
model_mix = load_model("/lfstev/deepskies/luisals/binary_classification/train_mixed_sims/model/"
                      "weights.58.hdf5")

