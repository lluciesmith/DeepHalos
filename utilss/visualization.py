import numpy as np
import matplotlib.pyplot as plt


m = load_model("/lfstev/deepskies/luisals/regression/train_mixed_sims/51_3_maxpool/model_100_epochs_mixed_sims.h5")

def visualize_kernels(model, num_layer=1):
    layers = model.layers
    filters, biases = layers[num_layer].get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_filters = filters.shape[-1]
    z_range = filters.shape[-3]
    ix = 1

    for i in range(n_filters):
        f = filters[:, :, :, 0, i]
        for j in range(z_range):
            ax = plt.subplot(n_filters, z_range, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("z = " + str(j))
            if i == 3:
                if j == 1:
                    ax.text(1, 4, "x")

            if i == 2:
                if j == 0:
                    ax.text(-2, -0.5, "y")
            plt.imshow(f[:, :, j], cmap='gray', vmin=0, vmax=1)
            ix += 1

    plt.subplots_adjust(bottom=0.14, top=0.94)
