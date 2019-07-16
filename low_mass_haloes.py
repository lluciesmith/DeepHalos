import numpy as np
import data_processing as dp
import CNN
import tensorflow
from tensorflow import set_random_seed
from tensorflow.keras.utils import plot_model
import pickle
import matplotlib.pyplot as plt


path = "/content/drive/My Drive/"

training_ids = np.load(path + "training_ids.npy")
validation_ids_training_sim = np.load(path + "validation_ids.npy")

halo_mass = np.load("/home/lls/stored_files/halo_mass_particles.npy")
p_ids = np.where(halo_mass > 0)[0]
# scaler, normalised_mass = dp.normalise_output(halo_mass, take_log=True)

bins = 10**np.arange(np.log10(halo_mass[p_ids]).min(), 15, 0.15)
mid_bins = (bins[1:] + bins[:-1])/2

num_p, b = np.histogram(halo_mass[p_ids], bins=bins)
m_p = 823774878.546787
num_h = num_p * m_p/mid_bins
bins_valid = mid_bins[num_h > 100]

tr_ids = training_ids[halo_mass[training_ids] < bins_valid.max()]
val_ids = validation_ids_training_sim[halo_mass[validation_ids_training_sim] < bins_valid.max()]

scaler, normalised_mass = dp.normalise_output(halo_mass[tr_ids], take_log=True)

