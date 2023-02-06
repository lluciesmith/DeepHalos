"""

theoretical_profiles
====================

Functional forms of common profiles (NFW as an example)

"""

import numpy as np
import abc, sys
import scipy.special

# # abc compatiblity with Python 2 *and* 3:
# # https://stackoverflow.com/questions/35673474/using-abc-abcmeta-in-a-way-it-is-compatible-both-with-python-2-7-and-python-3-5
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class AbstractBaseProfile(ABC):
    """
    Base class to generate functional form of known profiles. The class is organised a dictionary: access the profile
    parameters through profile.keys().

    To define a new profile, create a new class inheriting from this base class and define your own profile_functional()
    method. The static version can be handy to avoid having to create and object every time.
    As a example, the NFW functional is implemented.

    A generic fitting function is provided. Given profile data, e.g. quantity as a function of radius, it uses standard
    least-squares to fit the given functional form to the data.

    """
    def __init__(self):
        self._parameters = dict()

    @abc.abstractmethod
    def profile_functional(self, radius):
        pass

    @staticmethod
    @abc.abstractmethod
    def profile_functional_static(radius, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def jacobian_profile_functional_static(radius, **kwargs):
        """ Analytical expression of the jacobian of the profile for more robust fitting."""
        pass

    @classmethod
    def fit(cls, radial_data, profile_data, profile_err=None, use_analytical_jac=None, guess=None, log=False, bounds=None):
        """ Fit profile data with a leastsquare method.

        * profile_err * Error bars on the profile data as a function of radius. Can be a covariance matrix.
        * guess * Provide a list of parameters initial guess for optimisation
        """

        import scipy.optimize as so
        # Check data is not corrupted. Some are likely check in curve-fit already
        if np.isnan(radial_data).any() or np.isnan(profile_data).any():
            raise RuntimeError("Provided data contains NaN values")

        if np.count_nonzero(radial_data) != radial_data.size or np.count_nonzero(profile_data) != profile_data.size:
            raise RuntimeError("Provided data contains zeroes. This is likely to make the fit fail.")

        if radial_data.size != profile_data.size != profile_err.size:
            raise RuntimeError("Provided data arrays do not match in shape")

        if use_analytical_jac is not None:
            use_analytical_jac = cls.jacobian_profile_functional_static

        print(bounds)
        if bounds is None:
            profile_lower_bound = np.amin(profile_data)
            profile_upper_bound = np.amax(profile_data)
            radial_lower_bound = np.amin(radial_data)
            radial_upper_bound = np.amax(radial_data)
            if cls.num_params() > 2:
                bounds = ([profile_lower_bound, radial_lower_bound, 0.], [profile_upper_bound, radial_upper_bound, 0.5])
            else:
                bounds = ([profile_lower_bound, radial_lower_bound], [profile_upper_bound, radial_upper_bound])
        print(bounds)
        if log is True:
            function = cls.log_profile_functional_static
            err = [profile_err / (profile_data * np.log(10)) if profile_err is not None else None][0]
            data = np.log10(profile_data)
            if use_analytical_jac is not None:
                use_analytical_jac = lambda x, p1, p2: (cls.jacobian_profile_functional_static(x, p1, p2).transpose()
                                                        /(cls.profile_functional_static(x, p1, p2) * np.log(10))).transpose()

        else:
            function = cls.profile_functional_static
            err = profile_err
            data = profile_data

        try:
            print(function)
            parameters, cov = so.curve_fit(function,
                                           radial_data,
                                           data,
                                           sigma=err,
                                           p0=guess,
                                           bounds=bounds,
                                           check_finite=True,
                                           jac=use_analytical_jac,
                                           method='trf',
                                           ftol=1e-14,
                                           xtol=1e-14,
                                           gtol=1e-14,
                                           x_scale=1.0,
                                           loss='linear',
                                           f_scale=1.0,
                                           max_nfev=None,
                                           diff_step=None,
                                           tr_solver=None,
                                           verbose=2)
        except so.OptimizeWarning as w:
            raise RuntimeError(str(w))

        if (guess is None and any(parameters == np.ones(parameters.shape))) or any(parameters == guess):
            raise RuntimeError("Fitted parameters are equal to their initial guess. This is likely a failed fit.")

        return parameters, cov

    def __getitem__(self, item):
        return self._parameters.__getitem__(item)

    def __setitem__(self, key, value):
        raise KeyError('Cannot change a parameter from the profile once set')

    def __delitem__(self, key):
        raise KeyError('Cannot delete a parameter from the profile once set')

    def __repr__(self):
        return "<" + self.__class__.__name__ + str(list(self.keys())) + ">"

    def keys(self):
        return list(self._parameters.keys())

    @staticmethod
    def chi_squared(observed, expected, err=None, log=True):
        if log:
            observed = np.log10(observed)
            expected = np.log10(expected)
        if err is None:
            err = np.sqrt(expected)
        return np.sum((observed - expected) ** 2 / err ** 2)


class NFWprofile(AbstractBaseProfile):

    def __init__(self, halo_radius, scale_radius=None, density_scale_radius=None, concentration=None,
                 halo_mass=None):
        """
        To initialise an NFW profile, we always need:

          *halo_radius*: outer boundary of the halo (r200m, r200c, rvir ... depending on definitions)

        The profile can then be initialised either through scale_radius + central_density or halo_mass + concentration

          *scale_radius*: radius at which the slope is equal to -2

          *density_scale_radius*: 1/4 of density at r=rs (normalisation).

          *halo_mass*: mass enclosed inside the outer halo radius

          *concentration*: outer_radius / scale_radius

        From one mode of initialisation, the derived parameters of the others are calculated, e.g. if you initialise
        with halo_mass + concentration, the scale_radius and central density will be derived.

        """

        super(NFWprofile, self).__init__()

        self._halo_radius = halo_radius

        if scale_radius is None or density_scale_radius is None:
            if concentration is None or halo_mass is None or halo_radius is None:
                raise ValueError("You must provide concentration, virial mass"
                                 " if not providing the central density and scale_radius")
            else:
                self._parameters['concentration'] = concentration
                self._halo_mass = halo_mass

                self._parameters['scale_radius'] = self._derive_scale_radius()
                self._parameters['density_scale_radius'] = self._derive_central_overdensity()

        else:
            if concentration is not None or halo_mass is not None:
                raise ValueError("You can't provide both scale_radius+central_overdensity and concentration")

            self._parameters['scale_radius'] = scale_radius
            self._parameters['density_scale_radius'] = density_scale_radius

            self._parameters['concentration'] = self._derive_concentration()
            self._halo_mass = self.get_enclosed_mass(halo_radius)

    ''' Define static versions for use without initialising the class'''
    @staticmethod
    def profile_functional_static(radius, density_scale_radius, scale_radius):
        # Variable number of argument abstract methods only works because python is lazy with checking.
        # Is this a problem ?
        return density_scale_radius / ((radius / scale_radius) * (1.0 + (radius / scale_radius)) ** 2)

    @staticmethod
    def jacobian_profile_functional_static(radius, density_scale_radius, scale_radius):
        d_scale_radius = density_scale_radius * (3 * radius / scale_radius + 1) / (radius * (1 + radius / scale_radius) ** 3)
        d_central_density = 1 / ((radius / scale_radius) * (1 + radius / scale_radius) ** 2)
        return np.transpose([d_central_density, d_scale_radius])

    @staticmethod
    def log_profile_functional_static(radius, density_scale_radius, scale_radius):
        return np.log10(NFWprofile.profile_functional_static(radius, density_scale_radius, scale_radius))

    @staticmethod
    def get_dlogrho_dlogr_static(radius, scale_radius):
        return - (1.0 + 3.0 * radius / scale_radius) / (1.0 + radius / scale_radius)

    @staticmethod
    def num_params():
        return 2

    ''' Class methods'''
    def profile_functional(self, radius):
        return NFWprofile.profile_functional_static(radius, self._parameters['density_scale_radius'],
                                                    self._parameters['scale_radius'])

    def get_enclosed_mass(self, radius_of_enclosure):
        # Eq 7.139 in M vdB W
        return self._parameters['density_scale_radius'] * self._parameters['scale_radius'] ** 3 \
               * NFWprofile._helper_function(self._parameters['concentration'] *
                                             radius_of_enclosure / self._halo_radius)

    def _derive_concentration(self):
        return self._halo_radius / self._parameters['scale_radius']

    def _derive_scale_radius(self):
        return self._halo_radius / self._parameters['concentration']

    def _derive_central_overdensity(self):
        return self._halo_mass / (NFWprofile._helper_function(self._parameters['concentration'])
                                  * self._parameters['scale_radius'] ** 3)

    def get_dlogrho_dlogr(self, radius):
        return NFWprofile.get_dlogrho_dlogr_static(radius, self._parameters['scale_radius'])

    @staticmethod
    def _helper_function(x):
        return 4 * np.pi * (np.log(1.0 + x) - x / (1.0 + x))


class EinastoProfileFitAlpha(AbstractBaseProfile):

    def __init__(self, halo_radius, scale_radius=None, density_scale_radius=None, alpha=None,
                 concentration=None, halo_mass=None):
        """
        To initialise an Einasto profile, we always need:

          *halo_radius*: outer boundary of the halo (r200m, r200c, rvir ... depending on definitions)

        The Einasto profile class can be initialized from its fundamental
        parameters -- scale_radius, density_scale_radius, alpha -- or via mass and concentration.

          *scale_radius*: radius at which the slope is equal to -2

          *density_scale_radius*: 1/4 of density at r=rs (normalisation).

          *halo_mass*: mass enclosed inside the outer halo radius

          *concentration*: outer_radius / scale_radius

        From one mode of initialisation, the derived parameters of the others are calculated.
        In the case of mass+concentration initialisation, alpha is determined automatically from
        the peak height using the formula of Gao et al. 2008, alpha = 0.155 + 0.0095 * (peak_height)**2

        """

        super(EinastoProfileFitAlpha, self).__init__()

        self._halo_radius = halo_radius

        if scale_radius is None or density_scale_radius is None or alpha is None:
            if concentration is None or halo_mass is None or halo_radius is None:
                raise ValueError("You must provide concentration, virial mass"
                                 " if not providing the central density and scale_radius")
            else:
                self._parameters['concentration'] = concentration
                self._halo_mass = halo_mass

                self._parameters['scale_radius'] = self._derive_scale_radius()
                self._parameters['density_scale_radius'] = self._derive_central_overdensity()
                self._parameters['alpha'] = self._derive_alpha()

        else:
            if concentration is not None or halo_mass is not None:
                raise ValueError("You can't provide both scale_radius+central_overdensity and concentration")

            self._parameters['scale_radius'] = scale_radius
            self._parameters['density_scale_radius'] = density_scale_radius
            self._paramters['alpha'] = alpha

            self._parameters['concentration'] = self._derive_concentration()
            self._halo_mass = self.get_enclosed_mass(halo_radius)

    ''' Define static versions for use without initialising the class'''
    @staticmethod
    def profile_functional_static(radius, density_scale_radius, scale_radius, alpha):
        return density_scale_radius * np.exp(-2.0 / alpha * ((radius / scale_radius)**alpha - 1.0))

    @staticmethod
    def jacobian_profile_functional_static(radius, density_scale_radius, scale_radius, alpha):
        x = radius/scale_radius
        d_scale_radius = (2 * density_scale_radius * np.exp(-2.0 * (x**alpha - 1.0) / alpha) * x**alpha) / scale_radius
        d_central_density = np.exp(-2.0 / alpha * (x**alpha - 1.0))
        d_alpha = density_scale_radius * np.exp(-2.0 / alpha * (x**alpha - 1.0)) * \
                  ((2.0 / alpha**2 * (x**alpha - 1.0)) - (2.0 * x**alpha * np.log(x) / alpha))
        return np.transpose([d_central_density, d_scale_radius, d_alpha])

    @staticmethod
    def log_profile_functional_static(radius, density_scale_radius, scale_radius, alpha):
        return np.log10(EinastoProfileFitAlpha.profile_functional_static(radius, density_scale_radius, scale_radius, alpha))

    @staticmethod
    def get_dlogrho_dlogr_static(radius, scale_radius, alpha):
        return -2.0 * (radius / scale_radius)**alpha

    @staticmethod
    def num_params():
        return 3

    ''' Class methods'''
    def profile_functional(self, radius):
        return self.profile_functional_static(radius, self._parameters['density_scale_radius'],
                                                        self._parameters['scale_radius'], self._parameters['alpha'])

    def get_enclosed_mass(self, radius_of_enclosure):
        m_norm, gamma_3alpha = self._helper_mass_terms()
        gamma_inc = scipy.special.gammainc(3.0 / self._parameters['alpha'], 2.0 / self._parameters['alpha'] *
                                           (radius_of_enclosure / self._parameters['scale_radius']) ** self._parameters['alpha'])
        return m_norm * gamma_3alpha * gamma_inc

    def _derive_concentration(self):
        return self._halo_radius / self._parameters['scale_radius']

    def _derive_scale_radius(self):
        return self._halo_radius / self._parameters['concentration']

    def _derive_central_overdensity(self):
        return self._halo_mass / (NFWprofile._helper_function(self._parameters['concentration'])
                                  * self._parameters['scale_radius'] ** 3)
    @staticmethod
    def _derive_alpha():
        #peak_height = peaks.peakHeight(Mvir, z)
        #alpha = 0.155 + 0.0095 * (peak_height)**2
        return 0.16

    def get_dlogrho_dlogr(self, radius):
        return self.get_dlogrho_dlogr_static(radius, self._parameters['scale_radius'], self._parameters['alpha'])

    def _helper_mass_terms(self):
        _mass_norm = np.pi * self._parameters['rhos'] * self._parameters['rs'] ** 3 * \
                          2.0 ** (2.0 - 3.0 / self._parameters['alpha']) \
                          * self._parameters['alpha'] ** (-1.0 + 3.0 / self._parameters['alpha']) \
                          * np.exp(2.0 / self._parameters['alpha'])
        _gamma_3alpha = scipy.special.gamma(3.0 / self.par['alpha'])
        return _mass_norm, _gamma_3alpha


class EinastoProfileFixAlpha(EinastoProfileFitAlpha):

    def __init__(self, halo_radius, scale_radius=None, density_scale_radius=None, alpha=None,
                 concentration=None, halo_mass=None):

        super(EinastoProfileFixAlpha, self).__init__(halo_radius, scale_radius=scale_radius,
                                                     density_scale_radius=density_scale_radius, alpha=alpha,
                                                     concentration=concentration, halo_mass=halo_mass)

    @staticmethod
    def profile_functional_static(radius, density_scale_radius, scale_radius):
        alpha = EinastoProfileFixAlpha._derive_alpha()
        return density_scale_radius * np.exp(-2.0 / alpha * ((radius / scale_radius)**alpha - 1.0))

    @staticmethod
    def jacobian_profile_functional_static(radius, density_scale_radius, scale_radius):
        alpha = EinastoProfileFixAlpha._derive_alpha()
        x = radius/scale_radius
        d_scale_radius = (2 * density_scale_radius * np.exp(-2.0 * (x**alpha - 1.0) / alpha) * x**alpha) / scale_radius
        d_central_density = np.exp(-2.0 / alpha * (x**alpha - 1.0))
        d_alpha = density_scale_radius * np.exp(-2.0 / alpha * (x**alpha - 1.0)) * \
                  ((2.0 / alpha**2 * (x**alpha - 1.0)) - (2.0 * x**alpha * np.log(x) / alpha))
        return np.transpose([d_central_density, d_scale_radius, d_alpha])

    @staticmethod
    def log_profile_functional_static(radius, density_scale_radius, scale_radius):
        return np.log10(EinastoProfileFixAlpha.profile_functional_static(radius, density_scale_radius, scale_radius))

    @staticmethod
    def get_dlogrho_dlogr_static(radius, scale_radius):
        alpha = EinastoProfileFixAlpha._derive_alpha()
        return EinastoProfileFitAlpha.get_dlogrho_dlogr_static(radius, scale_radius, alpha)

    @staticmethod
    def num_params():
        return 2
