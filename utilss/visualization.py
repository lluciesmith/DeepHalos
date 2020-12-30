import numpy as np
from tensorflow.keras.models import load_model
import dlhalos_code.loss_functions as lf
import matplotlib.pyplot as plt
import scipy.stats
import tensorflow as tf

# m = load_model("/lfstev/deepskies/luisals/regression/train_mixed_sims/51_3_maxpool/model_100_epochs_mixed_sims.h5")


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


def weights_model(model):
    layers = model.layers
    conv_layers = [1, 5, 9, 13, 17, 23, 27]
    weights = []
    for i, l in enumerate(conv_layers):
        layer = layers[l]
        print(layer)
        filters, biases = layer.get_weights()
        weights.append(filters.flatten())
    return weights


def compare_reg_with_likelihood(model, generator_training, alpha=0.0005, norm="l1"):
    p = model.predict(generator_training, verbose=1)
    t = np.array([generator_training.labels[ID] for ID in generator_training.list_IDs])
    t = t.reshape(len(t), 1)
    t = tf.cast(t, tf.float32)
    gamma = model.layers[-1].get_weights()[0][0]
    gamma = tf.cast(gamma, tf.float32)
    loss_lik_ = lf.cauchy_selection_loss_fixed_boundary(gamma=gamma)(t, p)
    with tf.compat.v1.Session() as sess:
        loss_lik = sess.run(loss_lik_)
    loss_reg = reg_term(model, alpha=alpha, norm=norm)
    l_tot = model.evaluate(generator_training)
    return loss_lik, loss_reg, l_tot


def reg_term(model, alpha=0.0005, norm="l1"):
    layer_names_list = [layr.name for layr in model.layers]
    layers_conv3d = [s for s in layer_names_list if 'conv3d' in s]
    layers_dense = [s for s in layer_names_list if 'dense' in s]
    layers = layers_conv3d + layers_dense
    matched_indices = [i for i, item in enumerate(layer_names_list) if item in layers]
    weights_layers = [model.layers[index].get_weights()[0].flatten() for index in matched_indices]
    if norm == "l1":
        term = [alpha * np.sum(abs(w)) for w in weights_layers]
    else:
        term = [alpha * np.sum(w**2) for w in weights_layers]
    return term


def weight_evolution(params_model):
    model_epoch = load_model(params_model['path_model'] + "model/weights." + "05" + ".hdf5")
    w_i = weights_model(model_epoch)
    epochs = [5*i for i in range(2, 10)]
    epochs = np.array(epochs).astype('str')
    delta_weights = np.zeros((len(epochs), 7, 2))
    for i, epoch in enumerate(epochs):
        model_epoch = load_model(params_model['path_model'] + "model/weights." + epoch + ".hdf5")
        w_ii = weights_model(model_epoch)
        for l in range(len(w_ii)):
            delta_weights[i, l, 0] = np.mean((w_ii[l] - w_i[l])/w_i[l])
            delta_weights[i, l, 1] = np.std((w_ii[l] - w_i[l]) / w_i[l])
        w_i = w_ii
    return delta_weights


def get_weights_biases_layers(model, epoch, path_model):
    model.load_weights(path_model + 'model/weights.' + epoch + '.hdf5')
    # layer_names_list = [layr.name for layr in model.layers]
    # layers_conv3d = [s for s in layer_names_list if 'conv3d' in s]
    # layers_dense = [s for s in layer_names_list if 'dense' in s]
    # layers = layers_conv3d + layers_dense
    # matched_indices = [i for i, item in enumerate(layer_names_list) if item in layers]
    labels = ["conv0", "conv1", "conv2", "conv3", "conv4", "dense0", "dense1"]
    # layers = model.layers[matched_indices]
    l = [model.layers[i] for i in [1, 3, 6, 9, 12, 16, 19]]
    for i, layer in enumerate(l):
        print(layer, labels[i])
        filters, biases = layer.get_weights()
        np.save(path_model + 'weights/' + labels[i] + '_weights_epoch' + epoch + '.npy', filters)
        np.save(path_model + 'weights/' + labels[i] + '_biases_epoch' + epoch + '.npy', biases)