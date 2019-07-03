import matplotlib.pyplot as plt


def plot_true_vs_predict(true, predicted):
    f = plt.figure()
    plt.plot(true, true, color="dimgrey")
    plt.scatter(true, predicted, s=1)
    plt.xlabel("True log mass")
    plt.ylabel("Predicted log mass")
    plt.subplots_adjust(bottom=0.14)
    return f


def plot_loss(history, val_data=True):
    epochs = history.epoch
    loss_training = history.history["loss"]
    fig = plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_training, label="training")

    if val_data is True:
        loss_val = history.history["val_loss"]
        plt.plot(epochs, loss_val, label="validation")

    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel(history.model.loss, fontsize=18)
    plt.subplots_adjust(bottom=0.14)
    plt.legend(loc="best", fontsize=15)
    return fig


def plot_metric(history, val_data=True, ylabel="mae"):
    epochs = history.epoch
    mae_training = history.history["mean_absolute_error"]

    fig = plt.figure()
    plt.plot(epochs, mae_training, label="training")

    if val_data is True:
        mae_val = history.history["val_mean_absolute_error"]
        plt.plot(epochs, mae_val, label="validation")

    plt.xlabel("Epoch")
    if history.model.metrics_names[1] == "mean_absolute_error":
        plt.ylabel("mae")
    else:
        plt.ylabel(ylabel)
    plt.subplots_adjust(bottom=0.14)
    plt.legend(loc="best")
    return fig