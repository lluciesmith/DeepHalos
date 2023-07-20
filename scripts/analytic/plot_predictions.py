import pickle

# CNN raw density predictions
path = '/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/lr5e-5/'
p_raw = np.load(path + "seed_20/predicted_sim_6_epoch_09.npy")
t_raw = np.load(path + "seed_20/true_sim_6_epoch_09.npy")
diff_raw = p_raw - t_raw

path_av = "/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4.3/"
p_av = np.load(path_av + "predicted_sim_6_epoch_32.npy")
t_av = np.load(path_av + "true_sim_6_epoch_32.npy")
diff_av = p_av - t_av

# Analytic predictions
path_data = "/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/"
l = pickle.load(open(path_data + 'larger_labels_validation_set.pkl', 'rb'))
ids = np.array([int(ID[ID.find('-id-') + len('-id-'):]) for ID in list(l.keys())])

t = np.load("/Users/luisals/Projects/DLhalos/avg_densities/seed14/true_sim_6_epoch_32.npy")
eps_pred = np.load("/Users/luisals/Projects/DLhalos/analytic/EPS_predictions.npy")
diff_eps = np.log10(eps_pred[ids][eps_pred[ids] > 0]) - t[eps_pred[ids] > 0]

st_pred = np.load("/Users/luisals/Projects/DLhalos/analytic/ST_predictions.npy")
diff_st = np.log10(st_pred[ids][st_pred[ids] > 0]) - t[st_pred[ids] > 0]

print("KL between raw-density model and EPS residuals is %.2f" % kld.get_KL_div(diff_raw, diff_eps, 0.1, xlow=-3, xhigh=3)[0])
print("KL between raw-density model and ST residuals is %.2f" % kld.get_KL_div(diff_raw, diff_st, 0.1, xlow=-3, xhigh=3)[0])
print("KL between raw-density model and avg-density residuals is %.2f" % kld.get_KL_div(diff_raw, diff_av, 0.1, xlow=-3, xhigh=3)[0])

# plot residual distributions of which you are taking KL divergence
res_distr = [(diff_eps, "EPS"), (diff_st, "ST"), (diff_av, "avg-CNN")]
for resi in res_distr:
    _ = plt.hist(res_distr[0], bins=40, density=True, alpha=0.4, label=res_distr[1] + " residuals")
    _ = plt.hist(diff_raw, bins=40, density=True, alpha=0.4, label="CNN residuals", color="k")
    plt.axvline(x=0, color="dimgrey", ls="--")
    plt.legend(loc="best")
    plt.xlabel(r"$\log(M_\mathrm{pred}/M_\mathrm{true})$", fontsize=22)
    plt.subplots_adjust(bottom=0.17)
    plt.savefig("/Users/luisals/Projects/DLhalos/analytic/residuals_CNN_vs_" + res_distr[1] + ".png")


# plot 2D histograms of predictions
plt.hist2d(t[eps_pred > 0], np.log10(eps_pred[eps_pred>0]), bins=60)
plt.plot([11, 13.5], [11, 13.5], color="dimgrey", ls="--")
plt.xlabel("Truth")
plt.ylabel("EPS predictions")

plt.figure()
plt.hist2d(t[st_pred > 0], np.log10(st_pred[st_pred > 0]), bins=60)
plt.plot([11, 13.5], [11, 13.5], color="dimgrey", ls="--")
plt.xlabel("Truth")
plt.ylabel("ST predictions")


# contours
m = np.load("/Users/luisals/Projects/DLhalos/analytic/reseed6_halo_mass_particles.npy")
levels = (0.68, 0.95, 0.997)
bins = 50
idx_st = (m >= 10**11) & (m <= 10**13.4) & (st_pred > 0)
fig, ((ax3, ax4), (ax1, ax2)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6.9*1.3, 5.2*1.3))
# f = corner.hist2d(np.log10(m[idx_st]), np.log10(st_pred[idx_st]), levels=levels,
#                        bins=bins, smooth=True, color="chocolate", ax=ax1)
# idx_eps = (m >= 10**11) & (m <= 10**13.4) & (eps_pred > 0)
# f2 = corner.hist2d(np.log10(m[idx_eps]), np.log10(eps_pred[idx_eps]), levels=levels,
#                    bins=bins, smooth=True, color="darkred", ax=ax2)
f = corner.hist2d(t[st_pred[ids] > 0], np.log10(st_pred[ids][st_pred[ids] > 0]), levels=levels,
                       bins=bins, smooth=True, color="#1E3888", ax=ax1)
ax1.set_title("Sheth-Tormen")
idx_eps = (m >= 10**11) & (m <= 10**13.4) & (eps_pred > 0)
f2 = corner.hist2d(t[eps_pred[ids] > 0], np.log10(eps_pred[ids][eps_pred[ids] > 0]), levels=levels,
                   bins=bins, smooth=True, color="#47A8BD", ax=ax2)
ax2.set_title("Press-Schechter")
f11 = corner.hist2d(t_raw[t_raw >= 11], p_raw[t_raw >= 11], levels=levels,
                    bins=bins, smooth=True, color="#53131E", ax=ax3)
ax3.set_title("CNN raw-density")
f22 = corner.hist2d(t_av[t_av >= 11], p_av[t_av >= 11], levels=levels,
                    bins=bins, smooth=True, color="#B1740F", ax=ax3)
ax4.set_title("CNN avg-density")
xmin, xmax = plt.xlim()
for ax in [ax1, ax2, ax3, ax4]:
    ax.plot([xmin, xmax], [xmin, xmax], color="k")
plt.subplots_adjust(wspace=0.1, hspace=0.2, top=0.92, left=0.12, bottom=0.13)
fig.text(0.5, 0.02, r"$\log(M_\mathrm{true}/\mathrm{M_\odot})$", ha='center')
fig.text(0.02, 0.5, r"$\log(M_\mathrm{predicted}/\mathrm{M_\odot})$", va='center', rotation='vertical')


m = np.load("/Users/luisals/Projects/DLhalos/analytic/reseed6_halo_mass_particles.npy")
levels = (0.68, 0.95)
bins = 50
idx_st = (m >= 10**11) & (m <= 10**13.4) & (st_pred > 0)
fig, (ax3, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(14, 5.2))
f = corner.hist2d(t[st_pred[ids] > 0], np.log10(st_pred[ids][st_pred[ids] > 0]), levels=levels,
                       bins=bins, smooth=True, color="#1E3888", ax=ax1, contour_kwargs={'alpha':0.5})
f21 = corner.hist2d(t[st_pred[ids] > 0], np.log10(st_pred[ids][st_pred[ids] > 0]), levels=[levels[0]],
                   bins=bins, smooth=True, color="#1E3888", ax=ax1)
ax1.set_title("Sheth-Tormen")
idx_eps = (m >= 10**11) & (m <= 10**13.4) & (eps_pred > 0)
f2 = corner.hist2d(t[eps_pred[ids] > 0], np.log10(eps_pred[ids][eps_pred[ids] > 0]), levels=levels,
                   bins=bins, smooth=True, color="#47A8BD", ax=ax2, contour_kwargs={'alpha':0.5})
f22 = corner.hist2d(t[eps_pred[ids] > 0], np.log10(eps_pred[ids][eps_pred[ids] > 0]), levels=[levels[0]],
                   bins=bins, smooth=True, color="#47A8BD", ax=ax2)
ax2.set_title("Press-Schechter")
f11 = corner.hist2d(t_raw[t_raw >= 11], p_raw[t_raw >= 11], levels=levels,
                    bins=bins, smooth=True, color="#53131E", ax=ax3, contour_kwargs={'alpha':0.5})
f221 = corner.hist2d(t_raw[t_raw >= 11], p_raw[t_raw >= 11], levels=[levels[0]],
                   bins=bins, smooth=True, color="#53131E", ax=ax3)
ax3.set_title("CNN raw-density")
xmin, xmax = plt.xlim()
for ax in [ax1, ax2, ax3]:
    ax.plot([xmin, xmax], [xmin, xmax], color="dimgrey")
plt.subplots_adjust(wspace=0.1, hspace=0.2, top=0.92, left=0.08, bottom=0.1)
ax1.set_xlabel(r"$\log(M_\mathrm{true}/\mathrm{M_\odot})$")
ax3.set_ylabel(r"$\log(M_\mathrm{predicted}/\mathrm{M_\odot})$")
plt.ylim(10.4, 13.5)
plt.xlim(10.98, 13.4)



m = np.load("reseed6_halo_mass_particles.npy")
idx_eps = (m > 0) & (eps_pred > 0)
kld.get_KL_div(np.log10(m[idx_eps]), np.log10(eps_pred[idx_eps]), 0.1, xlow=10, xhigh=16)

idx_st = (m > 0) & (st_pred > 0)
kld.get_KL_div(np.log10(m[idx_st]), np.log10(st_pred[idx_st]), 0.1, xlow=10, xhigh=16)

diff_raw = p_raw - t_raw

path_av = "/Users/luisals/Documents/deep_halos_files/mass_range_13.4/random_20sims_200k/averaged_boxes/log_alpha_-4.3/"
p_av = np.load(path_av + "predicted_sim_6_epoch_32.npy")
t_av = np.load(path_av + "true_sim_6_epoch_32.npy")
diff_av = p_av - t_av

kl_raw_av = kde.get_KL_div(diff_raw, diff_av, 0.1, xlow=-3, xhigh=3)
print(kl_raw_av)


f, ax = plt.subplots()
f11 = corner.hist2d(t_raw[t_raw >= 11], p_raw[t_raw >= 11], alpha=0.3,
                    bins=70, smooth=True, color="#53131E", ax=ax, label="raw-density CNN")
f22 = corner.hist2d(t_av[t_av >= 11], p_av[t_av >= 11],
                    bins=70, smooth=True, color="#B1740F", ax=ax, label="avg-density CNN")
ax.plot([11, 13.4], [11, 13.4], ls="--", color="dimgrey")
ax.plot([], [], color="#53131E", label="raw-density CNN")
ax.plot([], [], color="#B1740F", label="avg-density CNN")
plt.legend(loc="best")
plt.ylim(10, 13.8)
ax.set_xlabel(r"$\log(M_\mathrm{true}/\mathrm{M_\odot})$")
ax.set_ylabel(r"$\log(M_\mathrm{predicted}/\mathrm{M_\odot})$")

f, ax = plt.subplots()
f11 = plt.hist2d(t_raw[t_raw >= 11], p_raw[t_raw >= 11], alpha=0.3,
                    bins=70, color="#53131E", label="raw-density CNN")
f22 = plt.hist2d(t_av[t_av >= 11], p_av[t_av >= 11],
                    bins=70, color="#B1740F", label="avg-density CNN")
ax.plot([11, 13.4], [11, 13.4], ls="--", color="dimgrey")
ax.plot([], [], color="#53131E", label="raw-density CNN")
ax.plot([], [], color="#B1740F", label="avg-density CNN")
plt.legend(loc="best")
plt.ylim(10, 13.8)
ax.set_xlabel(r"$\log(M_\mathrm{true}/\mathrm{M_\odot})$")
ax.set_ylabel(r"$\log(M_\mathrm{predicted}/\mathrm{M_\odot})$")