import sys
sys.path.append('/Users/lls/Documents/mlhalos_code/')
sys.path.append('/Users/lls/Documents/Projects/LightGBM_halos/')
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import predictions_functions as pf

path = "/Users/lls/Documents/deep_halos_files/regression/z0"
old_truth = np.load(path + "/truth1.npy")
old_pred = np.load(path + "/pred1_80.npy")

p1 = np.load(path + "/res75/pred1_80.npy")
t1 = np.load(path + "/res75/truth1.npy")

bins_violin = np.linspace(old_truth.min(), old_truth.max(), 14)

col1 = "#8C4843"
col2 = "#406D60"
col1_violin = "#A0524D"
col2_violin = "#55917F"
fig, ax = pf.compare_two_violin_plots(old_pred, old_truth, p1, t1,
                                      bins_violin, label1="$51^3$", label2="$75^3$",
                            col1=col1, col2=col2, col1_violin=col1_violin, col2_violin=col2_violin,
                            alpha1=0.3, alpha2=0.3, figsize=(6.9, 5.2),
                            edge1=col1, edge2=col2)
plt.legend(loc='lower right', bbox_to_anchor=(0.01, 0.48, 0.5, 0.5), framealpha=1)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
plt.subplots_adjust(left=0.14)
plt.savefig("/Users/lls/Documents/Papers/regression_paper1/violins_den_vs_shear.pdf")