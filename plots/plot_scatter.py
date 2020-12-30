import numpy as np
import matplotlib.pyplot as plt


def true_vs_predicted(predictions, truth, label=None, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(truth, predictions, label=label, s=0.2)
    ax.plot([10.5, 15], [10.5, 15], color="grey")

    ax.set_xlim(10.3, 15.1)
    ax.set_ylim(10.3, 15.1)

    ax.set_xlabel(r"$\log(M_\mathrm{true}/M_{\odot})$")
    ax.set_ylabel(r"$\log(M_\mathrm{predicted}/M_{\odot})$")

    if title is not None:
        ax.set_title(title)
        plt.subplots_adjust(bottom=0.14, top=0.9)
    else:
        plt.subplots_adjust(bottom=0.14)
    if label is not None:
        plt.legend(loc="best")
    return ax


