import sys
import numpy as np
import pynbody


def power_spectrum_from_CAMB_WMAP5(snapshot, save=True, path="/share/hypatia/lls/simulations/", omcdm=0.234, omb=0.045):
    import camb

    if path is None:
        path = "/Users/luisals/Projects/DLhalos/newruns/analytic/"

    maxkh=1E+02
    h = snapshot.properties['h']
    H0 = h * 100
    omch2 = omcdm * (h**2)
    ombh2 = omb * (h**2)
    assert omcdm + omb == snapshot.properties['omegaM0']

    # Define new parameters instance with WMAP5 cosmological parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    pars.set_matter_power(kmax=maxkh, k_per_logint=None, silent=False)
    pars.InitPower.set_params(ns=0.96)
    pars.validate()

    # Now get linear matter power spectrum at redshift 0

    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1E-04, maxkh=maxkh, npoints=150)
    powerspec = np.column_stack((kh, pk[0]))

    if save is True:
        np.savetxt(path + "camb_Pk_WMAP5", powerspec)
    else:
        return kh, pk[0]


def get_power_spectrum(snapshot, camb_path="/share/hypatia/lls/simulations/"):
    try:
        powerspec = pynbody.analysis.hmf.PowerSpectrumCAMB(snapshot, filename=camb_path + "camb_Pk_WMAP5")
        print("WARNING: Used CAMB saved power spectrum in" + str(camb_path) + "/camb_Pk_WMAP5")
    except IOError:
        print("WARNING: Save power spectrum not found - Computing and saving a new power spectrum in " + str(camb_path))
        power_spectrum_from_CAMB_WMAP5(snapshot, save=True, path=camb_path)
        powerspec = pynbody.analysis.hmf.PowerSpectrumCAMB(snapshot, filename=camb_path + "camb_Pk_WMAP5")
    powerspec.set_sigma8(0.817)
    print("sigma8 is " + str(powerspec.get_sigma8()))
    return powerspec


def get_variance(smoothing_scales, window_function, power_spectrum, snapshot):
    """ Mass needs to be in (Msol h^-1) units and radius needs to be in (Mpc h^-1 * a) units """
    if smoothing_scales.units == "Msol":
        h = snapshot.properties['h']

        smoothing_scales = smoothing_scales * h
        smoothing_scales.units = "Msol h**-1"

        arg_is_R = False

    elif smoothing_scales.units == "Msol h^-1":
        arg_is_R = False

    elif smoothing_scales.units == "Mpc":
        h = snapshot.properties['h']
        a = snapshot.properties['a']

        smoothing_scales = smoothing_scales * h / a
        smoothing_scales.units = "Mpc h**-1 a"

        arg_is_R = True

    elif smoothing_scales.units == "Mpc h**-1 a":
        arg_is_R = True

    elif smoothing_scales.units == "Mpc**-1 a**-1 h":
        smoothing_scales = 2*np.pi/smoothing_scales
        arg_is_R = True

    else:
        raise NameError("Select either radius/wavenumber or mass smoothing scales")

    var = pynbody.analysis.hmf.variance(smoothing_scales, window_function, power_spectrum, arg_is_R=arg_is_R)
    return var


def calculate_variance(smoothing_scales, snapshot, filter=None):
    powerspec = get_power_spectrum(snapshot)
    print("Done getting power spectrum")
    if filter is None:
        filter = powerspec._default_filter

    var = get_variance(smoothing_scales, filter, powerspec, snapshot)
    return var


def get_spherical_collapse_barrier(ics_snapshot=None, z=99, delta_sc_0=1.686, output="delta"):
    if z != 0:
        D_a = pynbody.analysis.cosmology.linear_growth_factor(ics_snapshot, z=z)
        delta_sc = delta_sc_0 * D_a
    elif z == 0:
        delta_sc = delta_sc_0
    else:
        raise AttributeError("Insert an integer for the redshift")
    if output == "rho/rho_bar":
        delta_sc += 1
    return delta_sc


def get_ellipsoidal_barrier_from_variance(variance, snapshot=None, z=99, beta=0.485, gamma=0.6, a=0.707,
                                          output="delta", delta_sc=None, delta_sc_0=1.686):
    if delta_sc is None:
        delta_sc = get_spherical_collapse_barrier(snapshot, z=z, output="delta", delta_sc_0=delta_sc_0)

    #var_squared = variance**2
    delta_sc_squared = delta_sc**2

    # a acts as a rescaler so one will recover B ~ delta_sc as the variance tends to zero only if a=1.
    B = np.sqrt(a) * delta_sc * (1 + (beta * ((variance / (a * delta_sc_squared))**gamma)))
    if output == "rho/rho_bar":
        B += 1
    return B

def ellipsoidal_collapse_barrier(mass_smoothing_scales, snapshot, beta=0.485, gamma=0.615, a=0.707, z=99,
                                 output="rho/rho_bar", delta_sc_0=1.686, filter=None):
    # beta = 0.47
    # a = 0.75 is favoured by Sheth & Tormen (2002) since in agreement
    # with Jenkins et al. (2001) halo mass function.

    variance = calculate_variance(mass_smoothing_scales, snapshot, filter=filter)
    B = get_ellipsoidal_barrier_from_variance(variance, snapshot, z=z, beta=beta, gamma=gamma, a=a, output=output,
                                              delta_sc_0=delta_sc_0)
    return B


if __name__ == "__main__":
    path = "/share/hypatia/lls/simulations/standard_reseed22/"
    ics = pynbody.load(path + 'IC.gadget2')
    ics.physical_units()

    m = np.logspace(10, 15, 50)
    m = pynbody.array.SimArray(m, 'Msol')

    bsph = ab.get_spherical_collapse_barrier(ics, z=99, delta_sc_0=1.686, output="rho/rho_bar")
    bell = ab.ellipsoidal_collapse_barrier(m, ics, z=99, delta_sc_0=1.686, output="rho/rho_bar")
