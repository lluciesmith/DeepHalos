import numpy as np
from plots import plots_for_predictions as pp
import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
from mlhalos import distinct_colours as dc
import matplotlib.pyplot as plt
from plots import plot_violins as pv

c = dc.get_distinct(4)

path = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'
p1 = np.load(path + "seed_20/predicted_sim_6_epoch_09.npy")
t1 = np.load(path + "seed_20/true_sim_6_epoch_09.npy")

f = pv.plot_violin(t1[t1>=11], p1[t1>=11], bins_violin=None, labels=None, return_stats=None, box=False,
                   alpha=0.8, vert=True, col=c[1], figsize=(8, 8))
plt.savefig("/Users/lls/Documents/Papers/dlhalos_paper/violin_z99_no_box.pdf")