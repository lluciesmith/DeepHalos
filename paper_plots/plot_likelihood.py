import numpy as np
from plots import plots_for_predictions as pp
from utilss import distinct_colours as dc
import matplotlib.pyplot as plt
from pickle import load

c = dc.get_distinct(6)

path = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'
p1 = np.load(path + "seed_20/all_predicted_sim_6_epoch_09.npy")
t1 = np.load(path + "seed_20/all_true_sim_6_epoch_09.npy")
g = np.load(path + "seed_20/gamma.npy")[9]

path2 = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4/'
scaler_training_set = load(open(path2 + 'scaler_output.pkl', 'rb'))
slices = [-0.85, -0.6, 0, 0.5, 0.75, 0.95]

f, a = pp.plot_likelihood_distribution(p1, t1, g, scaler_training_set, bins=None, fig=None, axes=None, color=c[4],
                                       title=None, legend=True, slices=slices)