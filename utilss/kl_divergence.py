import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from sklearn.neighbors import KernelDensity

# Fit a kde


def fit_kde_kernel(samples, bandwidth=0.1):
    ks = KernelDensity(bandwidth=bandwidth, atol=10 ** -10, rtol=10 ** -10)
    ks.fit(samples.reshape(-1, 1))
    return ks


def get_log_pdf_kde(kernel, x_grid=None):
    if x_grid is None:
        x_grid = np.linspace(10, 16, 500)
    log_pdf = kernel.score_samples(x_grid.reshape(-1, 1))
    return log_pdf


def get_log_pdf_from_samples(samples, bandwidth=0.1, x_grid=None):
    ks = fit_kde_kernel(samples, bandwidth=bandwidth)
    return get_log_pdf_kde(ks, x_grid=x_grid)


def run_kde_and_save(samples, bandwidth=0.23, saving_path="./kde.npy"):
    kde_samples = get_log_pdf_from_samples(samples, bandwidth=bandwidth)
    np.save(saving_path, kde_samples)


# KL Divergence


def f(fitted_kernel, x):
    x1 = np.array(x).reshape(1, -1)
    scoresx = fitted_kernel.score_samples(x1)
    return np.exp(scoresx)


def integrand(kernel1, kernel2, x):
    p = f(kernel1, x)
    q = f(kernel2, x)
    return p[p != 0] * np.log(p[p != 0] / q[p != 0])


def KL_div_continuous(kernel1, kernel2, xlow=9, xhigh=16):
    func = lambda x: integrand(kernel1, kernel2, x)
    return integrate.quad(func, xlow, xhigh)


def get_difference_in_kl_array2_array3_wrt_array1(array1_log_mass, array2_log_mass, array3_log_mass, bandwidth):
    kl_12 = get_KL_div(array1_log_mass, array2_log_mass, bandwidth)
    kl_13 = get_KL_div(array1_log_mass, array3_log_mass, bandwidth)
    return kl_12[0] - kl_13[0]


def get_KL_div(array1_log_mass, array2_log_mass, bandwidth):
    kde_1 = fit_kde_kernel(array1_log_mass, bandwidth=bandwidth)
    kde_2 = fit_kde_kernel(array2_log_mass, bandwidth=bandwidth)
    kl_1_vs_2 = KL_div_continuous(kde_1, kde_2)
    return kl_1_vs_2


# Plot functions

def plot_histogram(samples, bins=None, title=None, ylabel="PDF ground truth", figsize=(10, 6),
                   color="darkgrey", label=None):
    if bins is None:
        bins = np.linspace(10, 15, 201)
    ntruth, btruth = np.histogram(samples, bins=bins, density=True)
    bmid_truth = (btruth[1:] + btruth[:-1]) / 2

    plt.figure(figsize=figsize)
    plt.plot(bmid_truth, ntruth, color=color, label=label)

    plt.xlabel(r"$\log(M/\mathrm{M_\odot})$")
    plt.ylabel(ylabel)
    plt.legend(fontsize=13)
    plt.subplots_adjust(bottom=0.14)
    # plt.ylim(0,2)
    if title is not None:
        plt.title(title)
        plt.subplots_adjust(bottom=0.14, top=0.9)


def plot_kde_distribution(log_pdf, x_grid=None, label1="KDE", color=None, lw=1, ls="-"):
    if x_grid is None:
        x_grid = np.linspace(10, 16, 500)
    if color is not None:
        plt.plot(x_grid, np.exp(log_pdf), label=label1, color=color, lw=lw, ls=ls)
    else:
        plt.plot(x_grid, np.exp(log_pdf), label=label1)


