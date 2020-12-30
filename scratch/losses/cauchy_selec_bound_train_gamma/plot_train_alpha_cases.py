import numpy as np
import matplotlib.pyplot as plt
import re


def read_training_log_file(filepath):
    with open(filepath) as f:
        h = f.readlines()[1:]
    test = [re.sub(r'"', '', hi) for hi in h]
    test = [re.sub(r'\[', '', hi) for hi in test]
    test = [re.sub(r'\]', '', hi) for hi in test]
    test = [re.sub(r'\n', '', hi) for hi in test]
    for i, line in enumerate(test):
        test[i] = [float(elem) for elem in line.split(",")]
    return np.array(test)

path = "/Users/lls/Documents/deep_halos_files/regression/test_lowmass/reg_10000_perbin/larger_net" \
       "/cauchy_selec_gamma_bound/group_reg_alpha/train_alpha/"
cases = ["l2_conv_l1_dense_wdropout/new/", "l2_conv_l21_l1_dense/", "l2_conv_l21_l1_dense_wdropout/new/"]
labels = ["L2 (conv) + L1 (dense) + dropout", "L2 (conv) + L1 (dense) + L1-g (dense)",
          "L2 (conv) + L1 (dense) +  L1-g (dense) + dropout"]

# cases = ["l2_conv_l21_l1_dense/lr/0.001"]

#f, axes = plt.subplots(len(cases), 3, sharex=True, figsize=(13, 8))
f, axes = plt.subplots(len(cases), 3, sharex=True, figsize=(13, 8))
color = ["C" + str(i) for i in range(len(cases))]

for i, case in enumerate(cases):

    ax = axes[0, i]
    tr = read_training_log_file(path + case + "/training.log")
    ax.plot(tr[:,0], tr[:,1], lw=1.5, color=color[i])
    ax.plot(tr[:, 0], tr[:, 3], ls="--", color=color[i], lw=1.5)
    # ax.legend(loc="best", fontsize=14)
    if i == 0:
        ax.set_ylabel('Loss')
    ax.set_title(labels[i], fontsize=14)

    # p = np.load(path + case + "/trained_loss_params.npy")
    ep = np.insert(tr[:, 0], 0, tr[0, 0] - 1)

    ax = axes[1, i]
    g = np.load(path + case + "/trained_loss_gamma.npy")
    ax.plot(ep, g, color=color[i])
    ax.axhline(y=0.1, color="k", ls="--")
    ax.axhline(y=0.4, color="k", ls="--")
    if i == 0:
        ax.set_ylabel(r'$\gamma$')

    ax = axes[2, i]
    a = np.load(path + case + "/trained_loss_alpha.npy")
    ax.plot(ep, a, color=color[i])
    ax.axhline(y=-3, color="k", ls="--")
    ax.axhline(y=-4, color="k", ls="--")
    if i == 0:
        ax.set_ylabel(r'$\log_{10} (\alpha)$')

plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.08, bottom=0.1, top=0.95)
# plt.subplots_adjust(wspace=0.4, hspace=0.5, left=0.12, bottom=0.15, top=0.98)
f.text(0.5, 0.01, "Epoch")

for j in range(3):
    if j == 0:
        [axes[j, i].set_yscale("log") for i in range(3)]
    [axes[j, i].yaxis.set_major_formatter(plt.NullFormatter()) for i in range(1, 3)]
    ax0_lim = np.concatenate([list(axes[j, i].get_ylim()) for i in range(3)])
    [axes[j, i].set_ylim(ax0_lim.min(), ax0_lim.max()) for i in range(3)]


############ PLOT DIFFERENT LEARNING RATES ###############

path = "/Users/lls/Documents/deep_halos_files/regression/test_lowmass/reg_10000_perbin/larger_net" \
       "/cauchy_selec_gamma_bound/group_reg_alpha/train_alpha/l2_conv_l21_l1_dense/"
cases = [".", "lr/0.0005/", "lr/0.001/"]
labels = [r"lr$_\mathrm{init} = 10^{-4}$", r"lr$_\mathrm{init} = 5\times 10^{-4}$", "lr$_\mathrm{init} = 10^{-3}$"]

f, axes = plt.subplots(3, len(cases), sharex=True, figsize=(13, 8))
color = ["C" + str(i) for i in range(len(cases))]

for i, case in enumerate(cases):
    print(i)
    ax = axes[0, i]
    tr = read_training_log_file(path + case + "/training.log")
    ax.plot(tr[:,0], tr[:,1], lw=1.5, color=color[i], label="step decay")
    ax.plot(tr[:, 0], tr[:, 3], ls="--", color=color[i], lw=1.5)
    ax.legend(loc="best", fontsize=14)
    if i == 0:
        ax.set_ylabel('Loss')
    ax.set_title(labels[i], fontsize=14)

    # p = np.load(path + case + "/trained_loss_params.npy")
    ep = np.insert(tr[:, 0], 0, tr[0, 0] - 1)

    ax = axes[1, i]
    g = np.load(path + case + "/trained_loss_gamma.npy")
    ax.plot(ep, g, color=color[i])
    ax.axhline(y=0.1, color="k", ls="--")
    ax.axhline(y=0.4, color="k", ls="--")
    if i == 0:
        ax.set_ylabel(r'$\gamma$')

    ax = axes[2, i]
    a = np.load(path + case + "/trained_loss_alpha.npy")
    ax.plot(ep, a, color=color[i])
    ax.axhline(y=-3, color="k", ls="--")
    ax.axhline(y=-4, color="k", ls="--")
    if i == 0:
        ax.set_ylabel(r'$\log_{10} (\alpha)$')

plt.subplots_adjust(wspace=0.2, hspace=0.15, left=0.08, bottom=0.1, top=0.95)
f.text(0.5, 0.01, "Epoch")

i = 1
color = "C3"
case1 = cases[i] + "no_decay/"
ax = axes[0, i]
tr = read_training_log_file(path + case1 + "/training.log")
ax.plot(tr[:, 0], tr[:, 1], lw=1.5, color=color, label="no decay")
ax.plot(tr[:, 0], tr[:, 2], ls="--", color=color, lw=1.5)
ax.legend(loc="best", fontsize=14)

# p = np.load(path + case + "/trained_loss_params.npy")
ep = np.insert(tr[:, 0], 0, tr[0, 0] - 1)

ax = axes[1, i]
g = np.load(path + case1 + "/trained_loss_gamma.npy")
ax.plot(ep, g, color=color)

ax = axes[2, i]
a = np.load(path + case1 + "/trained_loss_alpha.npy")
ax.plot(ep, a, color=color)

i = 1
color = "C4"
case2 = cases[i] + "exp_decay/"
ax = axes[0, i]
tr = read_training_log_file(path + case2 + "/training.log")
ax.plot(tr[:, 0], tr[:, 1], lw=1.5, color=color, label="exp decay")
ax.plot(tr[:, 0], tr[:, 2], ls="--", color=color, lw=1.5)
ax.legend(loc="best", fontsize=14)

# p = np.load(path + case + "/trained_loss_params.npy")
ep = np.insert(tr[:, 0], 0, tr[0, 0] - 1)

ax = axes[1, i]
g = np.load(path + case2 + "/trained_loss_gamma.npy")
ax.plot(ep, g, color=color)

ax = axes[2, i]
a = np.load(path + case2 + "/trained_loss_alpha.npy")
ax.plot(ep, a, color=color)


########### PLOT GRID SEARCH IN LOG-ALPHA ##################

path_all = "/Users/lls/Documents/deep_halos_files/regression/test_lowmass/reg_10000_perbin/larger_net/" \
          "cauchy_selec_gamma_bound/group_reg_alpha/train_alpha/"
reg_types = ["l2_conv_l21_l1_dense/",
             # "l2_conv_l1_dense_wdropout/", "l2_conv_l21_l1_dense_wdropout/"
             ]
paths = [path_all + reg_type for reg_type in reg_types]

for j, path in enumerate(paths):
    reg_type_i = reg_types[j][:-2]
    log_alpha_values = [-3.1, -3.3, -3.5, -3.7, -3.9]
    cases = ["log_alpha_" + str(l) for l in log_alpha_values]
    labelss = [r"$\log(\alpha) = %.1f$" % l for l in log_alpha_values]

    f, axes = plt.subplots(len(cases), 2, sharex=True, figsize=(12, 8))
    color = ["C" + str(i) for i in range(len(cases))]

    for i, case in enumerate(cases):

        ax = axes[i, 0]
        tr = read_training_log_file(path + case + "/training.log")
        if i == 2:
            ax.plot(tr[1:, 0], tr[1:,1], lw=1.5, color=color[i], label=labelss[i])
            ax.plot(tr[1:, 0], tr[1:, 3], ls="--", color=color[i], lw=1.5)
        else:
            ax.plot(tr[:, 0], tr[:,1], lw=1.5, color=color[i], label=labelss[i])
            ax.plot(tr[:, 0], tr[:, 3], ls="--", color=color[i], lw=1.5)

        ax.legend(loc="best", fontsize=14)
        # ax.set_ylim(-1, 10)
        # ax.set_yscale("log")
        if i == 0:
            ax.set_title('Loss')

        ep = np.insert(tr[:, 0], 0, tr[0, 0] - 1)
        ax = axes[i, 1]
        g = np.load(path + case + "/trained_loss_gamma.npy")
        ax.plot(ep, g[:], color=color[i])
        # ax.axhline(y=0.1, color="grey", ls="--")
        # ax.axhline(y=0.4, color="grey", ls="--")
        ax.set_ylim(0.01, 0.41)
        if i == 0:
            ax.set_title(r'$\gamma$')

    plt.subplots_adjust(wspace=0.2, hspace=0, left=0.08, bottom=0.12, top=0.95)
    f.text(0.5, 0.01, "Epoch")
    plt.savefig(path + "grid_search_log_alpha_" + reg_type_i + ".png")

########### PLOT GRID SEARCH IN LOG-ALPHA ##################

path_all = "/Users/lls/Documents/deep_halos_files/full_mass_range/xavier/alpha/alpha"
alpha_values = ["-2", "-2.5","-3"]

f, axes = plt.subplots(len(alpha_values), 3, sharex=True, figsize=(14, 8))
color = ["C" + str(i) for i in range(len(alpha_values))]

for i in range(len(alpha_values)):

    path = path_all + alpha_values[i]
    label = r"$\log(\alpha) = $" + alpha_values[i]

    ax = axes[i, 0]
    tr = np.loadtxt(path + "/training.log", delimiter=",", skiprows=1)
    ax.plot(tr[:, 0], tr[:,2], lw=1.5, color=color[i], label=label)
    ax.plot(tr[:, 0], tr[:, 5], ls="--", color=color[i], lw=1.5)

    ax.legend(loc="best", fontsize=14)
    # ax.set_ylim(-1, 10)
    if i==0 or i==1:
        ax.set_yscale("log")
    if i == 0:
        ax.set_title('Loss')

    ax = axes[i, 1]
    tr = np.loadtxt(path + "/training.log", delimiter=",", skiprows=1)
    ax.plot(tr[:, 0], tr[:, 1], lw=1.5, color=color[i], label=label)
    ax.plot(tr[:, 0], tr[:, 4], ls="--", color=color[i], lw=1.5)
    if i == 0:
        ax.set_title('Likelihood')
    ax.set_ylim(-0.1, 0.5)


    ep = np.insert(tr[:, 0], 0, tr[0, 0] - 1)
    ax = axes[i, 2]
    g = np.loadtxt(path + "/gamma.txt", delimiter=",")
    ax.plot(ep, g[:], color=color[i])
    # ax.axhline(y=0.1, color="grey", ls="--")
    # ax.axhline(y=0.4, color="grey", ls="--")
    ax.set_ylim(0.01, 0.41)
    if i == 0:
        ax.set_title(r'$\gamma$')

plt.subplots_adjust(wspace=0.2, hspace=0, left=0.08, bottom=0.12, top=0.95)
f.text(0.5, 0.01, "Epoch")


path_all = "/Users/lls/Documents/deep_halos_files/restricted_mass_range/alpha"
alpha_values = ["-2", "-3"]

f, axes = plt.subplots(len(alpha_values), 3, sharex=True, figsize=(14, 8))
color = ["C" + str(i) for i in range(len(alpha_values))]

for i in range(len(alpha_values)):

    path = path_all + alpha_values[i]
    label = r"$\log(\alpha) = $" + alpha_values[i]

    ax = axes[i, 0]
    tr = np.loadtxt(path + "/training.log", delimiter=",", skiprows=1)
    ax.plot(tr[:, 0], tr[:,2], lw=1.5, color=color[i], label=label)
    ax.plot(tr[:, 0], tr[:, 5], ls="--", color=color[i], lw=1.5)

    ax.legend(loc="best", fontsize=14)
    # ax.set_ylim(-1, 10)
    #if i==0 or i==1:
    #    ax.set_yscale("log")
    if i == 0:
        ax.set_title('Loss')

    ax = axes[i, 1]
    tr = np.loadtxt(path + "/training.log", delimiter=",", skiprows=1)
    ax.plot(tr[:, 0], tr[:, 1], lw=1.5, color=color[i], label=label)
    ax.plot(tr[:, 0], tr[:, 4], ls="--", color=color[i], lw=1.5)
    if i == 0:
        ax.set_title('Likelihood')
    ax.set_ylim(-0.5, 1)


    ep = np.insert(tr[:, 0], 0, tr[0, 0] - 1)
    ax = axes[i, 2]
    # g = np.loadtxt(path + "/gamma.txt", delimiter=",")
    g = np.load(path + "/trained_loss_gamma.npy")
    #g = np.insert(g, 0, 0.2)
    ax.plot(ep, g[:], color=color[i])
    # ax.axhline(y=0.1, color="grey", ls="--")
    # ax.axhline(y=0.4, color="grey", ls="--")
    ax.set_ylim(0.01, 0.41)
    if i == 0:
        ax.set_title(r'$\gamma$')

plt.subplots_adjust(wspace=0.2, hspace=0, left=0.08, bottom=0.12, top=0.95)
f.text(0.5, 0.01, "Epoch")
