"""
:mod:`shear`

Computes the shear tensor at a smoothing radius scale.

The shear tensor field is a 3x3 matrix, given by the second derivative of the
gravitational potential. The gravitational potential is calculated by the Poisson
equation in Fourier space from the density field smoothed at a given radius
smoothing scale.

"""
import numpy as np
import scipy.linalg
import scipy.special
from scipy.constants import G
import scipy.linalg.lapack as la
from multiprocessing import Pool
import time


class Shear(object):
    """ Compute the shear tensor (3x3 array) at each grid point of the box (shape x shape x shape)"""

    def __init__(self, snapshot, smoothing_scale, number_of_processors=40, path=None):
        """
        Instantiates :class:`shear` given a snapshot and a smoothing scale.
        """

        self.snapshot = snapshot
        self.smoothing_scale = smoothing_scale.in_units(self.snapshot["x"].units)

        self.number_of_processors = number_of_processors
        self.path = path

        self.boxsize = self.snapshot.properties['boxsize'].in_units(self.snapshot["x"].units)
        self.shape = int(scipy.special.cbrt(len(self.snapshot)))

        self._shear_eigenvalues = None
        self._density_subtracted_eigenvalues = None

    @property
    def shear_eigenvalues(self):
        if self._shear_eigenvalues is None:
            self._shear_eigenvalues = self.calculate_shear_eigenvalues(self.snapshot, self.smoothing_scale)

        return self._shear_eigenvalues

    @property
    def density_subtracted_eigenvalues(self):
        if self._density_subtracted_eigenvalues is None:
            self._density_subtracted_eigenvalues = self.calculate_density_subtracted_eigenvalues()

        return self._density_subtracted_eigenvalues

    def get_smoothed_fourier_density(self, snapshot, smoothing_scale, box_shape, boxsize, path="/home/lls/"):
        density = snapshot.dm['rho'].reshape((box_shape, box_shape, box_shape))
        density_k_space = np.fft.fftn(density)

        top_hat = TopHatWindow(smoothing_scale, boxsize, box_shape, path=path)
        window_function = top_hat.top_hat_k_space

        den_smooth = window_function * density_k_space
        return den_smooth

    def rescale_density_to_density_contrast_in_fourier_space(self, density_fourier):
        den_contrast_fourier = density_fourier / (density_fourier[0, 0, 0] / (self.shape ** 3))
        den_contrast_fourier[0, 0, 0] = 0
        return den_contrast_fourier

    def get_k_coordinates_box(self, shape, boxsize):
        """ Return the k-coordinates of the simulation box """

        if shape == 256:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix.npy')
                else:
                    print("Loading Fourier transform matrix for shape " + str(shape))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix.npy")

            except IOError:
                print("Calculating Fourier transform matrix for shape " + str(shape))
                a = window.TopHat.grid_coordinates_for_fourier_transform(shape)

        elif shape == 512:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix_shape_512.npy')
                    print("loading Fourier transform matrix of shape " + str(shape))
                else:
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix_shape_512.npy")
                    print("loading Fourier transform matrix of shape " + str(shape))

            except IOError:
                print("Calculating Fourier transform matrix for shape " + str(shape))
                a = window.TopHat.grid_coordinates_for_fourier_transform(shape)

        else:
            print("Not using FT matrix")
            a = window.TopHat.grid_coordinates_for_fourier_transform(shape)

        k_coord = 2. * np.pi * a / boxsize
        return k_coord

    def get_norm_k_box(self, shape, boxsize):
        k_coord = self.get_k_coordinates_box(shape, boxsize)
        norm_squared = np.power(np.fabs(k_coord), 2)
        return norm_squared

    @staticmethod
    def poisson_equation_in_fourier_space(rho_contrast_fourier, norm_k_squared):
        """ Rescaled Poisson's equation in Fourier space """
        norm_k_squared[0, 0, 0] = 1

        phi = - rho_contrast_fourier / norm_k_squared
        phi[0, 0, 0] = 0
        return phi

    def get_potential_from_density_fourier_space(self, density, shape, boxsize):
        """
        Computes the gravitational potential give the density via Poisson's equation in Fourier space.

        """
        den_contrast = self.rescale_density_to_density_contrast_in_fourier_space(density)
        k_norm_squared = self.get_norm_k_box(shape, boxsize)
        potential = self.poisson_equation_in_fourier_space(den_contrast, k_norm_squared)
        return potential

    def shear_prefactor_ki_kj(self, i, j, k, k_mode):
        nyquist = int(self.shape/2)
        k_mode_coordinates = np.zeros((3, 3))

        k_mode_coordinates[0] = [k_mode[i] * k_mode[ijk] for ijk in [i, j, k]]
        k_mode_coordinates[1] = [k_mode[j] * k_mode[ijk] for ijk in [i, j, k]]
        k_mode_coordinates[2] = [k_mode[k] * k_mode[ijk] for ijk in [i, j, k]]

        if any([mode == nyquist for mode in [i, j, k]]):

            if i == nyquist and j != nyquist and k!= nyquist:
                position_nyquist = 0
                position_not_nyquist = [1, 2]

            elif i != nyquist and j == nyquist and k != nyquist:
                position_nyquist = 1
                position_not_nyquist = [0, 2]

            elif i != nyquist and j != nyquist and k == nyquist:
                position_nyquist = 2
                position_not_nyquist = [0, 1]

            elif i == nyquist and j == nyquist and k != nyquist:
                position_nyquist = [0, 1]
                position_not_nyquist = 2

            elif i == nyquist and j != nyquist and k == nyquist:
                position_nyquist = [0, 2]
                position_not_nyquist = 1

            else:
                position_nyquist = [1, 2]
                position_not_nyquist = 0

            k_mode_coordinates[position_nyquist, position_not_nyquist] = 0
            k_mode_coordinates[position_not_nyquist, position_nyquist] = 0
        else:
            pass

        return k_mode_coordinates

    def calculate_shear_in_single_grid(self, potential, i, j, k, k_mode):
        k_mode_coordinates = self.shear_prefactor_ki_kj(i, j, k, k_mode)
        shear_single_grid = - k_mode_coordinates * potential[i, j, k]

        return shear_single_grid

    def get_shear_from_potential_in_fourier_space(self, potential, shape, boxsize):
        """
        Computes the 3x3 shear tensor at each grid point from the gravitational potential.
        This function takes ~5.25 minutes to run on a single processor.
        """
        shape = int(shape)
        a_space = np.concatenate((np.arange(shape/2), np.arange(-shape/2, 0)))
        k_mode = 2 * np.pi * a_space / boxsize

        shear_fourier = np.zeros((shape, shape, shape, 3, 3), dtype=complex)

        for i in range(shape):
            for j in range(shape):
                for k in range(shape):

                    if i == 0 and j == 0 and k == 0:
                        pass

                    else:
                        shear_ijk = self.calculate_shear_in_single_grid(potential, i, j, k, k_mode)
                        shear_fourier[i, j, k] = shear_ijk

        return shear_fourier

    def get_shear_tensor_in_real_space_from_potential(self, potential, shape, boxsize):
        """
        Returns the 3x3 shear tensor in real space at each grid point,
        after computing the sheat tensor from the potential in Fourier space.

        The computation takes ~5.25 minutes to run and the inverse
        Fourier transform takes ~26 seconds.

        """
        shear_fourier = self.get_shear_from_potential_in_fourier_space(potential, shape, boxsize)
        shear_real = np.real(np.fft.ifftn(shear_fourier, axes=(0, 1, 2)).reshape((shape**3, 3, 3)))

        return shear_real

    def get_shear_tensor_at_scale(self, density_scale):
        """
        Shear tensor computation.
        """
        shape = self.shape
        boxsize = self.boxsize

        potential = self.get_potential_from_density_fourier_space(density_scale, shape, boxsize)
        shear_real = self.get_shear_tensor_in_real_space_from_potential(potential, shape, boxsize)

        return shear_real

    @staticmethod
    def get_eigenvalues_matrix(matrix):
        """
        Sort the eigenvalues such that eigval1 >= eigval2 >= eigval3.
        This function takes ~8 microseconds. It uses LAPACK Fortran package which is much faster
        than default scipy implementation (about ~10 times faster).
        """
        eig_real, eig_im, eigvec_real, eigvec_im, info = la.dgeev(matrix, compute_vl=0, compute_vr=0, overwrite_a=1)
        eig_sorted = np.sort(eig_real)[::-1]
        return eig_sorted

    def get_eigenvalues_many_matrices(self, matrices, number_of_processors=10):

        t00 = time.time()
        t0 = time.clock()

        if number_of_processors == 1:
            shear_eigenvalues = [self.get_eigenvalues_matrix(matrices[i]) for i in range(len(matrices))]

        else:
            pool = Pool(processes=number_of_processors)
            function = self.get_eigenvalues_matrix
            shear_eigenvalues = pool.map(function, matrices)
            pool.close()
            pool.join()

        print("Wall time " + str(time.time() - t00))
        print("Process time " + str(time.clock() - t0))

        return np.array(shear_eigenvalues)

    def get_eigenvalues_shear_tensor_from_density(self, density, number_of_processors=10):
        """
        Calculating the eigenvalues takes a long time.
        It is recommended to use multiprocessing - specify number of processors to use.

        """

        shear_tensor = self.get_shear_tensor_at_scale(density)
        shear_eigenvalues = self.get_eigenvalues_many_matrices(shear_tensor, number_of_processors=number_of_processors)
        return shear_eigenvalues

    def calculate_shear_eigenvalues(self, snapshot, smoothing_scale):
        density = self.get_smoothed_fourier_density(snapshot, smoothing_scale, self.shape, self.boxsize)
        eigvals = self.get_eigenvalues_shear_tensor_from_density(density, self.number_of_processors)
        return eigvals

    def get_sum_eigvals(self, eigvals):
        if eigvals.shape[1] == 3:
            sum_eig = np.sum(eigvals, axis=1)

        else:
            sum_eig = np.column_stack([np.sum(eigvals[:, (3 * i): (3 * i) + 3], axis=1)
                                       for i in range(len(self.shear_scale))])
        return sum_eig

    @staticmethod
    def subtract_a_third_of_trace(eigenvalues_with_trace, sum_eigenvalues):
        density_to_subtract = np.column_stack((sum_eigenvalues / 3, sum_eigenvalues / 3, sum_eigenvalues / 3))
        density_subtracted_eigenvalues = eigenvalues_with_trace - density_to_subtract
        return density_subtracted_eigenvalues

    def subtract_density_from_eigenvalues(self, eigenvalues_with_trace, sum_eigenvalues):
        if eigenvalues_with_trace.shape[1] == 3:
            density_to_subtract = np.column_stack((sum_eigenvalues / 3, sum_eigenvalues / 3, sum_eigenvalues / 3))
            d_sub_eigenvalues = eigenvalues_with_trace - density_to_subtract

        else:
            d_sub_eigenvalues = np.zeros((eigenvalues_with_trace.shape))

            for i in range(len(self.shear_scale)):
                eig_i = eigenvalues_with_trace[:, int(3 * i):int(3 * i) + 3]
                sum_i = sum_eigenvalues[:, i]

                d_sub_eigenvalues[:, int(3 * i):int(3 * i) + 3] = self.subtract_a_third_of_trace(eig_i,  sum_i)

        return d_sub_eigenvalues

    def calculate_density_subtracted_eigenvalues(self):
        eigenvalues = self.shear_eigenvalues
        sum_eigenvalues = self.get_sum_eigvals(eigenvalues)

        density_subtracted_eigenvalues = self.subtract_density_from_eigenvalues(eigenvalues, sum_eigenvalues)
        return density_subtracted_eigenvalues


class ShearProperties(Shear):

    def __init__(self, snapshot, smoothing_scale, number_of_processors=40, path=None):
        Shear.__init__(self, snapshot, smoothing_scale, number_of_processors=number_of_processors, path=path)

        self._ellipticity = None
        self._prolateness = None

        self._density_subtracted_ellipticity = None
        self._density_subtracted_prolateness = None

        self._sum_eigvals = None

    @property
    def ellipticity(self):
        if self._ellipticity is None:
            self._ellipticity = self.get_ellipticity(subtract_density=False)

        return self._ellipticity

    @property
    def prolateness(self):
        if self._prolateness is None:
            self._prolateness = self.get_prolateness(subtract_density=False)

        return self._prolateness

    @property
    def density_subtracted_ellipticity(self):
        if self._density_subtracted_ellipticity is None:
            self._density_subtracted_ellipticity = self.get_ellipticity(subtract_density=True)

        return self._density_subtracted_ellipticity

    @property
    def density_subtracted_prolateness(self):
        if self._density_subtracted_prolateness is None:
            self._density_subtracted_prolateness = self.get_prolateness(subtract_density=True)

        return self._density_subtracted_prolateness

    def get_ellipticity(self, subtract_density=False):
        # sort eigenvalues such that lambda_1 >= lambda_2 >= lambda_3
        if subtract_density is True:

            eigvals = self.density_subtracted_eigenvalues
            sum_ids = None

        else:
            eigvals = self.shear_eigenvalues
            sum_ids = self.get_sum_eigvals(eigvals)

        ellip = self.calculate_ellipticity(eigvals, sum_ids)
        return ellip

    def get_prolateness(self, subtract_density=False):
        if subtract_density is True:
            eigvals = self.density_subtracted_eigenvalues
            sum_ids = None

        else:
            eigvals = self.shear_eigenvalues
            sum_ids = self.get_sum_eigvals(eigvals)

        prol = self.calculate_prolateness(eigvals, sum_ids)
        return prol

    @staticmethod
    def calculate_ellipticity(eigenvalues, sum_ids=None):
        if sum_ids is None:
            ellipticity = (eigenvalues[:, 0] - eigenvalues[:, 2])

        else:
            ellipticity = (eigenvalues[:, 0] - eigenvalues[:, 2]) / (2 * sum_ids)

        return ellipticity

    @staticmethod
    def calculate_prolateness(eigenvalues, sum_ids=None):
        if sum_ids is None:
            prolateness = (3 * (eigenvalues[:, 0] + eigenvalues[:, 2]))

        else:
            prolateness = (eigenvalues[:, 0] + eigenvalues[:, 2] - (2 * (eigenvalues[:, 1]))) / (2 * sum_ids)

        return prolateness


class TopHatWindow:

    def __init__(self, smoothing_scale, boxsize, shape, path=None):
        self.path = path
        self.top_hat_k_space = self.top_hat_filter_in_k_space(smoothing_scale, boxsize, shape)

    def top_hat_filter_in_k_space(self, radius, boxsize, shape):
        """
        Defines Fourier top-hat filter function in simulation box (ndarray).

        Args:
            radius (SimArray): Radius of top-hat window function.
            boxsize (array): physical size of simulation box.
            shape (int): number of grids in box.

        Returns:
            top_hat_k_space (ndarray): Top-hat window function in Fourier space.

        """

        if shape == 256:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    # print("loading top hat filter matrix")
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix.npy')
                else:
                    print("loading top hat filter matrix")
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix.npy")

            except IOError:
                print("Calculating top hat filter matrix")
                a = self.grid_coordinates_for_fourier_transform(shape)

        elif shape == 512:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix_shape_512.npy')
                else:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix_shape_512.npy")

            except IOError:
                print("Calculating top hat filter matrix for shape " + str(shape))
                a = self.grid_coordinates_for_fourier_transform(shape)

        elif shape == 2048:
            try:
                # Load Fourier_transform_matrix if available containing grid coordinate values.
                if self.path is None:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load('/home/lls/stored_files/Fourier_transform_matrix_shape_2048.npy')
                else:
                    print("loading top hat filter matrix of shape " + str(shape))
                    a = np.load(self.path + "/stored_files/Fourier_transform_matrix_shape_2048.npy")

            except IOError:
                print("Calculating top hat filter matrix for shape " + str(shape))
                a = self.grid_coordinates_for_fourier_transform(shape)

        else:
            a = self.grid_coordinates_for_fourier_transform(shape)
        # Impose a[0, 0, 0] = 1 to avoid warnings for zero-division when evaluating top_hat.
        # Valid imposition since top_hat[0, 0, 0] need also to be 1.
        a[0, 0, 0] = 1

        k = 2. * np.pi * a / boxsize
        radius = float(radius)

        top_hat = (3. * (np.sin(k * radius) - ((k * radius) * np.cos(k * radius)))) / ((k * radius) ** 3)

        # we impose top_hat[0,0,0] = 1 since we want k=0 mode to be 1.
        top_hat[0, 0, 0] = 1.

        return top_hat

    @staticmethod
    def grid_coordinates_for_fourier_transform(shape):
        """Assigns coordinates to box grids (ndarray)."""

        a = np.zeros((shape, shape, shape))

        for i in range(shape):
            for j in range(shape):
                for k in range(shape):

                    if (i >= shape / 2) and (j >= shape / 2) and (k >= shape / 2):
                        a[i, j, k] = np.sqrt((i - shape) ** 2 + (j - shape) ** 2 + (k - shape) ** 2)

                    elif (i >= shape / 2) and (j >= shape / 2) and (k < shape / 2):
                        a[i, j, k] = np.sqrt((i - shape) ** 2 + (j - shape) ** 2 + k ** 2)

                    elif (i >= shape / 2) and (j < shape / 2) and (k >= shape / 2):
                        a[i, j, k] = np.sqrt((i - shape) ** 2 + j ** 2 + (k - shape) ** 2)

                    elif (i < shape / 2) and (j >= shape / 2) and (k >= shape / 2):
                        a[i, j, k] = np.sqrt(i ** 2 + (j - shape) ** 2 + (k - shape) ** 2)

                    elif (i >= shape / 2) and (j < shape / 2) and (k < shape / 2):
                        a[i, j, k] = np.sqrt((i - shape) ** 2 + j ** 2 + k ** 2)

                    elif (i < shape / 2) and (j >= shape / 2) and (k < shape / 2):
                        a[i, j, k] = np.sqrt(i ** 2 + (j - shape) ** 2 + k ** 2)

                    elif (i < shape / 2) and (j < shape / 2) and (k >= shape / 2):
                        a[i, j, k] = np.sqrt(i ** 2 + j ** 2 + (k - shape) ** 2)

                    else:
                        a[i, j, k] = np.sqrt(i ** 2 + j ** 2 + k ** 2)
        return a

    @staticmethod
    def Wk(kR):
        return (3. * (np.sin(kR) - ((kR) * np.cos(kR)))) / ((kR) ** 3)
