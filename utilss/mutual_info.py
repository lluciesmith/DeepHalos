import numpy as np
from sklearn.neighbors import KernelDensity
import scipy.integrate as integrate


def mutual_information_discrete(x, y, bins=100):
    """ Mutual information of two discrete variables """
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def mutual_information_cont(x, y, bandwidth=0.1, xlow=9, xhigh=16, ylow=9, yhigh=16):
    # KL divergence between joint probability distribution p(x, y) and
    pxy = kde2D(x, y, bandwidth)
    print("Fitted 2D kde to joint distributions")
    px = kde1D(x, bandwidth)
    py = kde1D(y, bandwidth)
    print("Fitted 1D kdes for each marginal distributions")
    return KL_div_continuous(pxy, px, py, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)


def kde1D(x, bandwidth, **kwargs):
    """Build 1D kernel density estimate (KDE)."""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl = kde_skl.fit(x.reshape(-1, 1))
    return kde_skl


def evaluate_kde1d(fitted_kde, xx):
    """Evaluate fitted 1D KDE at position xx."""
    z = np.exp(fitted_kde.score_samples(np.array([xx]).reshape(-1, 1)))
    return z


def kde2D(x, y, bandwidth, **kwargs):
    """Build 2D kernel density estimate (KDE)."""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(np.vstack([y, x]).T)
    return kde_skl


def evaluate_kde2d(fitted_kde, xx, yy):
    """Evaluate fitted 2D KDE at position xx."""
    xy_sample = np.vstack([yy, xx]).T
    z = np.exp(fitted_kde.score_samples(xy_sample))
    return z


def integrand(fitted_pxy, fitted_px, fitted_py, x, y):
    pxy = evaluate_kde2d(fitted_pxy, x, y)
    px = evaluate_kde1d(fitted_px, x)
    py = evaluate_kde1d(fitted_py, y)
    nsz = pxy != 0
    return pxy[nsz] * np.log(pxy[nsz] / (px[nsz] * py[nsz]))


def KL_div_continuous(fitted_pxy, fitted_px, fitted_py, xlow=9, xhigh=16, ylow=9, yhigh=16):
    func = lambda x, y: integrand(fitted_pxy, fitted_px, fitted_py, x, y)
    return integrate.dblquad(func, xlow, xhigh, lambda x: ylow, lambda x: yhigh)