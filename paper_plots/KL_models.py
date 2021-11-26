import numpy as np
from utilss import kl_divergence as kde

path = '/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'
p_raw = np.load(path + "seed_20/predicted_sim_6_epoch_09.npy")
t_raw = np.load(path + "seed_20/true_sim_6_epoch_09.npy")
diff_raw = p_raw - t_raw

path_av = "/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4.3/"
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

path_gbt = "/Users/luisals/Documents/mlhalos_files/LightGBM/CV_only_reseed_sim/"
truth_gbt = np.load(path_gbt + "/truth_shear_den_test_set.npy")
shear_gbt = np.load(path_gbt + "/predicted_shear_den_test_set.npy")
ran = np.random.choice(np.arange(len(shear_gbt)), 50000)
diff_gbt = truth_gbt - shear_gbt

kl_raw_gbt = kde.get_KL_div(diff_raw[(t_raw >= 11.4) & (t_raw <= 13.4)],
                            diff_gbt[ran][(truth_gbt[ran] >= 11.4) & (truth_gbt[ran] <= 13.4)], 0.1, xlow=-3, xhigh=3)
print(kl_raw_gbt)

kl_av_gbt = kde.get_KL_div(diff_av[(t_av >= 11.4) & (t_av <= 13.4)],
                            diff_gbt[ran][(truth_gbt[ran] >= 11.4) & (truth_gbt[ran] <= 13.4)], 0.1, xlow=-3, xhigh=3)
print(kl_av_gbt)


mass_bins = np.linspace(11, 13.4, 4, endpoint=True)
for i in range(len(mass_bins) - 1):
    idx = (t_raw >= mass_bins[i]) & (t_raw < mass_bins[i+1])
    kl_raw_av = kde.get_KL_div(diff_raw[idx], diff_av[idx], 0.1, xlow=-3, xhigh=3)
    print(kl_raw_av)

ids = np.load("/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/ids_larger_validation_set.npy")
r = np.load("/Users/luisals/Documents/mlhalos_files/reseed6/reseed6radius_in_halo_particles.npy")[ids]
r_vir = np.load("/Users/luisals/Documents/mlhalos_files/reseed6/reseed6_virial_radius_particles.npy")[ids]
frac_r = r / r_vir

# with bands
diff_seeds = [np.load(path + "seed_" + seed + "/predicted_sim_6_epoch_" + epoch + ".npy") - t_raw
              for seed, epoch in [("21", "09"), ("22", "08"), ("23", "08"), ("24", "09")]]
diff_av_seeds = [np.load("/Users/luisals/Projects/DLhalos/avg_densities/seed" + seed + "/predicted_sim_6_epoch_" + epoch + ".npy") - t_av
                 for seed, epoch in [("11", "32"), ("12", "35"), ("13", "33"), ("14", "40")]]
diff_av_seeds.append(p_av - t_av)

rbins = np.linspace(0.2, 1, 4)
mass_bins = np.linspace(t1[t1>=11].min(), t1[t1>=11].max() + 10**-6, 13)
bandwidth = 0.3

kls_av_seeds = np.zeros((len(mass_bins) - 1, len(rbins) - 1, len(diff_av_seeds)))
kls_raw_seeds = np.zeros((len(mass_bins) - 1, len(rbins) - 1, len(diff_seeds)))
for i in range(len(mass_bins) - 1):
    for j in range(len(rbins) - 1):
        idx = (t_raw >= mass_bins[i]) & (t_raw < mass_bins[i + 1]) & (frac_r >= rbins[j]) & (frac_r < rbins[j+1])
        print(len(t_raw[idx]))
        for k in range(len(diff_av_seeds)):
            kls_av_seeds[i, j, k] = kde.get_KL_div(diff_raw[idx], diff_av_seeds[k][idx], bandwidth, xlow=-3, xhigh=3)[0]
        for k in range(len(diff_seeds)):
            kls_raw_seeds[i, j, k] = kde.get_KL_div(diff_raw[idx], diff_seeds[k][idx], bandwidth, xlow=-3, xhigh=3)[0]

f, ax = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex="all", sharey="all")
for i in range(len(ax)):
    label = r"$%.1f\leq r/r_\mathrm{vir} \leq %.1f$" % (rbins[i], rbins[i+1])
    for k in range(len(diff_av_seeds)):
        ax[i].scatter((mass_bins[:-1] + mass_bins[1:])/2, kls_av_seeds[:, i, k], color="C"+ str(i), s=20)
    for k in range(len(diff_seeds)):
        ax[i].scatter((mass_bins[:-1] + mass_bins[1:])/2, kls_raw_seeds[:, i, k], color="C"+ str(i), marker="*", s=40)
    text(0.2, 0.8, label, horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes, fontsize=14)
    # ax[i].grid(axis="y")

plt.subplots_adjust(wspace=0, hspace=0, top=0.9)
plt.xlabel(r"$\log(M_\mathrm{true}/\mathrm{M_\odot})$", fontsize=16)
ax[1].set_ylabel("K-L divergence", fontsize=16)

ax[0].scatter([], [], marker="*", s=40, label="raw-density models w diff seeds", color="k")
ax[0].scatter([], [], marker=".", s=40, label="raw- vs avg-density models w diff seeds", color="k")
ax[0].legend(bbox_to_anchor=(0.47, 1.4), ncol=2, fontsize=13, loc='upper center')
plt.ylim(-0.05, 0.37)
[a.grid(axis='y') for a in ax]



kl_raw_av = [kde.get_KL_div(diff_raw, diff_av_seeds[i], 0.1, xlow=-3, xhigh=3)[0] for i in range(len(diff_av_seeds))]
kl_raw_raw = [kde.get_KL_div(diff_raw, diff_seeds[i], 0.1, xlow=-3, xhigh=3)[0] for i in range(len(diff_seeds))]

