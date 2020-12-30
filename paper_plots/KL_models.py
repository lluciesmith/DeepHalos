import numpy as np
import sys
sys.path.append('/Users/lls/Documents/Projects/LightGBM_halos/')
from utilss import kl_divergence as kde

path = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'
p_raw = np.load(path + "seed_20/predicted_sim_6_epoch_09.npy")
t_raw = np.load(path + "seed_20/true_sim_6_epoch_09.npy")
diff_raw = p_raw - t_raw

path_av = "/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4.3/"
p_av = np.load(path_av + "predicted_sim_6_epoch_32.npy")
t_av = np.load(path_av + "true_sim_6_epoch_32.npy")
diff_av = p_av - t_av

kl_raw_av = kde.get_KL_div(diff_raw, diff_av, 0.1, xlow=-3, xhigh=3)
print(kl_raw_av)

random_pred = np.copy(t_av)
np.random.shuffle(random_pred)
diff_random = random_pred - t_av
kl_worst = kde.get_KL_div(diff_raw, diff_random, 0.1, xlow=-3, xhigh=3)
print(kl_worst)

path_gbt = "/Users/lls/Documents/mlhalos_files/LightGBM/CV_only_reseed_sim/"
truth_gbt = np.load(path_gbt + "/truth_shear_den_test_set.npy")
shear_gbt = np.load(path_gbt + "/predicted_shear_den_test_set.npy")
ran = np.random.choice(np.arange(len(shear_gbt)), 50000)
diff_gbt = truth_gbt - shear_gbt

kl_raw_gbt = kde.get_KL_div(diff_raw[(t_raw >= 11.4) & (t_raw <= 13.4)],
                            diff_gbt[ran][(truth_gbt[ran] >= 11.4) & (truth_gbt[ran] <= 13.4)], 0.1, xlow=-3, xhigh=3)
print(kl_raw_gbt)