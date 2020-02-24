import numpy as np
import matplotlib.pyplot as plt
import re
import math
from mlhalos import parameters
import pynbody


def load_genpk(path,box):
    """Load a GenPk format power spectum, plotting the DM and the neutrinos (if present)
    Does not plot baryons."""
    #Load DM P(k)
    matpow=np.loadtxt(path)
    scale=2*math.pi/ box
    #Adjust Fourier convention to match CAMB.
    simk=matpow[1:, 0] * scale
    Pk=matpow[1:, 1] * box**3
    return (simk,Pk)

def plot_genpk_power(matpow1, box,color=None, ls="-", label=None):
    """ Plot the matter power as output by gen-pk"""
    (k, Pk1)=load_genpk(matpow1,box)
    #^2*2*!PI^2*2.4e-9*k*hub^3
    plt.ylabel(r"P($k$) /($h^{-3}$ Mpc$^{3}$)")
    plt.xlabel(r"$k /(h$ Mpc$^{-1}$)")
    plt.title("Power spectrum")
    if label is not None:
        plt.loglog(k, Pk1, linestyle=ls, color=color, label=label)
        plt.legend(loc="best")
    else:
        plt.loglog(k, Pk1, linestyle=ls, color=color)


if __name__ == "__main__":
    ic = parameters.InitialConditionsParameters(
        initial_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/IC.gadget3",
        final_snapshot="/Users/lls/Documents/mlhalos_files/reseed50/snapshot_099")

    paths = ["./nina_sim/PK-DM-snapshot_104", "./reseed_50/PK-DM-snapshot_099", "./reseed50_2/PK-DM-snapshot_099",
             "./reseed3/PK-DM-snapshot_099", "./reseed4/PK-DM-snapshot_099", "./reseed5/ PK - DM - snapshot_099",
             "./reseed6/PK-DM-snapshot_007", "./reseed7/PK-DM-snapshot_007", "./reseed8/PK-DM-snapshot_007",
             "./reseed9/PK-DM-snapshot_007", "./reseed10/PK-DM-snapshot_007"]

    labels = ["sim 0", "sim 1", "sim 2", "sim 3", "sim 4", "sim 5", "sim 6", "sim 7", "sim 8", "sim 9", "sim 10",
              "sim 11"]
    Pk = pynbody.analysis.hmf.PowerSpectrumCAMB(ic.final_snapshot)

    for i in range(len(paths)):
        plot_genpk_power(paths[i], 50, label=labels[i])

    plt.loglog(Pk.k, Pk(Pk.k), color="k")

