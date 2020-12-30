import numpy as np
import matplotlib.pyplot as plt

path = '/Users/lls/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'

t_0 = np.load(path + "seed_20/true_sim_6_epoch_09.npy")
p_0 = np.load(path + "seed_20/predicted_sim_6_epoch_09.npy")
p_1 = np.load(path + "seed_21/predicted_sim_6_epoch_09.npy")
p_2 = np.load(path + "seed_22/predicted_sim_6_epoch_08.npy")
p_3 = np.load(path + "seed_23/predicted_sim_6_epoch_08.npy")
p_4 = np.load(path + "seed_24/predicted_sim_6_epoch_09.npy")
preds = [p_0, p_1, p_2, p_3, p_4]

mass_bins = np.linspace(11, 13.4, 5, endpoint=True)
b00 = t_0 < mass_bins[0]
b0 = (t_0>=mass_bins[0]) & (t_0<=mass_bins[1])
b1 = (t_0 > mass_bins[1]) & (t_0<=mass_bins[2])
b2 = (t_0 > mass_bins[2]) & (t_0<=mass_bins[3])
b3 = (t_0 > mass_bins[3]) & (t_0<=mass_bins[4])

labels = [r"$\log M_\mathrm{true} \leq %.1f$" % mass_bins[0],
          r"$%.1f \leq \log M_\mathrm{true} \leq %.1f$" %(mass_bins[0], mass_bins[1]),
          r"$%.1f \leq \log M_\mathrm{true} \leq %.1f$" %(mass_bins[1], mass_bins[2]),
          r"$%.1f \leq \log M_\mathrm{true} \leq %.1f$" %(mass_bins[2], mass_bins[3]),
          r"$%.1f \leq \log M_\mathrm{true} \leq %.1f$" %(mass_bins[3], mass_bins[4])]

bins = np.linspace(-3, 3, 80)
for i, b in enumerate([b00, b0, b1, b2, b3]):
    plt.figure()
    for p in [p_0, p_1, p_2, p_3, p_4]:
        _ = plt.hist(p[b] - t_0[b], lw=1.2, bins=bins, alpha=0.7, histtype="step")
        plt.axvline(x=0, color="dimgrey")
        if np.allclose(b, b00):
            plt.title(labels[i], color="r")
        else:
            plt.title(labels[i])
        plt.xlabel(r"$\log (M_\mathrm{predicted}/M_\mathrm{true})$")
        plt.subplots_adjust(bottom=0.14, top=0.92, left=0.14)

    plt.savefig(path + "pred_multiple_seed_bin_" + str(i) + ".png")


m = np.linspace(t_0.min(), t_0.max(), 10, endpoint=True)
for j, p in enumerate([p_1, p_2, p_3, p_4], start=1):
    plt.figure()
    plt.scatter(t_0, p- p_0, s=0.06)
    plt.axhline(y=0, color="dimgrey")
    for i in range(len(mass_bins) - 1):
        plt.scatter(np.mean([m[i], m[i+1]]), np.mean((p - p_0)[(t_0>=m[i]) & (t_0 <m[i+1])]), s=10, color="k")
        plt.errorbar(np.mean([m[i], m[i+1]]), np.mean((p - p_0)[(t_0>=m[i]) & (t_0 <m[i+1])]),
                     yerr=np.std((p - p_0)[(t_0>=m[i]) & (t_0 <m[i+1])]), color="k", lw=2)
    plt.axvline(x=11, color="r", lw=1.5)
    plt.xlabel(r"$\log M_\mathrm{true}$")
    plt.ylabel(r"$\log (M_\mathrm{predicted}^{\mathrm{RUN} \, %i}/M_\mathrm{predicted})^{\mathrm{RUN} \, 0}$" %j)
    plt.subplots_adjust(bottom=0.14, left=0.18)
    plt.savefig(path + "difference_predictions_seeds_" + str(j) + ".png")


