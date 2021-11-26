import numpy as np
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath("./DeepHalos/"))))
from utilss import distinct_colours as dc
import matplotlib.pyplot as plt
from plots import plot_violins as pv
import seaborn as sns
from utilss import kl_divergence as kde

c = dc.get_distinct(4)

path = '/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'
p1 = np.load(path + "seed_20/predicted_sim_6_epoch_09.npy")
t1 = np.load(path + "seed_20/true_sim_6_epoch_09.npy")

f = pv.plot_violin(t1[t1>=11], p1[t1>=11], bins_violin=None, labels=None, return_stats=None, box=False,
                   alpha=0.8, vert=True, col=c[1], figsize=(8, 8))
plt.savefig("/Users/lls/Documents/Papers/dlhalos_paper/violin_z99_no_box.pdf")


p_pot = np.load('/Users/luisals/Projects/DLhalos/pot/predicted_sim_6_epoch_23.npy')
t_pot = np.load('/Users/luisals/Projects/DLhalos/pot/true_sim_6_epoch_23.npy')
f1 = pv.plot_violin([t1[t1 >= 11], t_pot[t_pot >= 11]], [p1[t1 >= 11], p_pot[t_pot >= 11]],
                    bins_violin=None, labels=["density", "potential"], return_stats=None, box=False,
                    alpha=[0.8, 0.5], vert=True, figsize=None)
xl = f1[1].get_xlim()
yl = f1[1].get_ylim()


# KL divergences
true_all = np.concatenate([t1[t1>=11], t_pot[t_pot>=11]])
bins_violin = np.linspace(true_all.min(), true_all.max() + 10**-6, 13)
bmid = (bins_violin[1:] + bins_violin[:-1])/2
seeds = ['21', '22', '23', '24']
kl_scatter = np.zeros((len(seeds), len(bins_violin)-1))

for k, n in enumerate(seeds):
    if n == '22' or n == '23':
        pn = np.load(path + "seed_" + str(n) + "/predicted_sim_6_epoch_08.npy")
        tn = np.load(path + "seed_" + str(n) + "/true_sim_6_epoch_08.npy")
    else:
        pn = np.load(path + "seed_" + str(n) + "/predicted_sim_6_epoch_09.npy")
        tn = np.load(path + "seed_" + str(n) + "/true_sim_6_epoch_09.npy")
    for i in range(len(bins_violin) - 1):
        p0 = pn[(tn >= bins_violin[i]) & (tn <= bins_violin[i+1])] - tn[(tn >= bins_violin[i]) & (tn <= bins_violin[i+1])]
        pd = p1[(t1 >= bins_violin[i]) & (t1 <= bins_violin[i + 1])] - t1[(t1 >= bins_violin[i]) & (t1 <= bins_violin[i + 1])]
        kl_scatter[k, i] = kde.get_KL_div(p0, pd, 0.15, xlow=-3, xhigh=3)[0]

kl_pot = [kde.get_KL_div(p_pot[(t_pot >= bins_violin[i]) & (t_pot <= bins_violin[i+1])] - t_pot[(t_pot >= bins_violin[i]) & (t_pot <= bins_violin[i+1])],
                         p1[(t1 >= bins_violin[i]) & (t1 <= bins_violin[i + 1])] - t1[(t1 >= bins_violin[i]) & (t1 <= bins_violin[i + 1])],
                         0.15, xlow=-3, xhigh=3)[0] for i in range(len(bins_violin) - 1)]

f, ax = plt.subplots(1, 1, figsize=(9, 4))
bmid_round = ['%.1f' % i for i in bmid]
c2 ="#FF6700"
c1="#090c9b"
[ax.scatter(bmid_round, kli_scatter, s=40, color=c2) for kli_scatter in kl_scatter]
ax.scatter(bmid_round, kl_pot, s=100, label="density vs potential", color=c1, marker="*")
ax.scatter([], [], label="density vs density w different seeds", s=40, color=c2)
ax.xaxis.grid(True)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel(r"$\log(M_\mathrm{true}/\mathrm{M_\odot})$", fontsize=16)
plt.ylabel("K-L divergence", fontsize=16)
plt.legend(loc="upper center", fontsize=14)
plt.subplots_adjust(left=0.1, bottom=0.2)

# plots

rpot = {"pred": np.concatenate((p_pot[t_pot >= 11], p1[t1 >= 11])),
        "true": bmid[np.digitize(true_all, bins_violin) - 1],
        "Trained on": np.concatenate((np.repeat("Gravitational potential", len(p_pot[t_pot >= 11])),
                                     np.repeat("Density field", len(p1[t1 >= 11]))))}
palette = {"Gravitational potential": c[0], "Density field": c[1]}
ax = sns.violinplot(x="true", y="pred", hue="Trained on", data=rpot, palette=palette,
                    split=True, inner=None, cut=0)
plt.plot([-1, 12], [bmid[0] - np.diff(bmid)[0], bmid[-1] + np.diff(bmid)[0]], color="k")
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = ['%.1f' % float(label) for label in labels]
ax.set_xticklabels(labels, fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlim(-0.5, 11.5)

plt.ylabel(r"$\log(M_\mathrm{predicted}/\mathrm{M_\odot})$", fontsize=16)
plt.xlabel(r"$\log(M_\mathrm{true}/\mathrm{M_\odot})$", fontsize=16)
plt.legend(loc="best", fontsize=14)
plt.subplots_adjust(left=0.14, bottom=0.13)

# plots
true_all = np.concatenate([t1[t1>=11], t_big[t_big>=11]])
bins_violin = np.linspace(true_all.min(), true_all.max() + 10**-6, 13)
bmid = (bins_violin[1:] + bins_violin[:-1])/2
rpot = {"pred": np.concatenate((p1[t1 >= 11], p_big[t_big >= 11])),
        "true": bmid[np.digitize(true_all, bins_violin) - 1],
        "Trained on": np.concatenate((np.repeat(r"$L_\mathrm{box}=200 \, \mathrm{Mpc} \,/ \,h$", len(t_big[t_big >= 11])),
                                     np.repeat(r"$L_\mathrm{box}=50 \, \mathrm{Mpc} \,/ \,h$", len(p1[t1 >= 11]))))}
palette = {r"$L_\mathrm{box}=200 \, \mathrm{Mpc} \,/ \,h$": c[0], r"$L_\mathrm{box}=50 \, \mathrm{Mpc} \,/ \,h$": c[1]}
ax = sns.violinplot(x="true", y="pred", hue="Trained on", data=rpot, palette=palette,
                    split=True, inner=None, cut=0)
plt.plot([-1, 12], [bmid[0] - np.diff(bmid)[0], bmid[-1] + np.diff(bmid)[0]], color="k")
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = ['%.1f' % float(label) for label in labels]
ax.set_xticklabels(labels, fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlim(-0.5, 11.5)

plt.ylabel(r"$\log(M_\mathrm{predicted}/\mathrm{M_\odot})$", fontsize=16)
plt.xlabel(r"$\log(M_\mathrm{true}/\mathrm{M_\odot})$", fontsize=16)
plt.legend(loc="best", fontsize=14)
plt.subplots_adjust(left=0.14, bottom=0.13)


