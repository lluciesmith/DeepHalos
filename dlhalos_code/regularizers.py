import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K


def l2_norm(alpha):
    return L2(alpha)


def l1_norm(alpha):
    return L1(alpha)


def l1_and_l21_group(alpha):
    return L21_and_L1(alpha)


def l21_group(alpha):
    return L21(alpha)


class RegClass:
    def __init__(self, alpha):
        self.alpha = K.variable(K.cast_to_floatx(alpha))

    def set_alpha(self, alpha):
        val_alpha = K.cast_to_floatx(alpha)
        K.set_value(self.alpha, val_alpha)


class L2(Regularizer, RegClass):
    """ Regularizer for combined L21 group regularization and L1 regularization. """

    def __init__(self, alpha=0.):
        super().__init__(alpha)

    def __call__(self, x):
        regularization = 0.
        regularization += self.alpha * K.sum(K.square(x))
        return regularization

    def get_config(self):
        return {'alpha_l2': float(self.alpha)}


class L1(Regularizer, RegClass):
    """ Regularizer for combined L21 group regularization and L1 regularization. """

    def __init__(self, alpha=0.):
        super().__init__(alpha)

    def __call__(self, x):
        regularization = 0.
        regularization += self.alpha * K.sum(K.abs(x))
        return regularization

    def get_config(self):
        return {'alpha_l1': self.alpha}


class L21_and_L1(Regularizer, RegClass):
    """ Regularizer for combined L21 group regularization and L1 regularization. """

    def __init__(self, alpha=0.):
        super().__init__(alpha)

    def __call__(self, x):
        regularization = 0.
        regularization += self.alpha * K.sum(K.abs(x))
        regularization += self.alpha * K.sum(K.sqrt(K.sum(K.square(x), axis=1)))
        return regularization

    def get_config(self):
        return {'alpha_l21_l1': self.alpha}


class L21(Regularizer, RegClass):
    """ Regularizer for L21 regularization. """

    def __init__(self, alpha=0.):
        super().__init__(alpha)

    def __call__(self, x):
        # We do not add the normalization coefficient for now
        # const_coeff = np.sqrt(K.int_shape(x)[1])

        regularization = 0.
        regularization += self.alpha * K.sum(K.sqrt(K.sum(K.square(x), axis=1)))
        return regularization

    def get_config(self):
        return {'alpha_l21': self.alpha}


def active_neurons(model):
    # Utility function to count active neurons in a Keras model with Dense layers
    layer_names_list = [layr.name for layr in model.layers]
    layers_dense = [s for s in layer_names_list if 'dense' in s]
    matched_indices = [i for i, item in enumerate(layer_names_list) if item in layers_dense]
    sum_neurons = lambda index: np.sum(np.abs(model.layers[index].get_weights()[0]), axis=1)
    sum_count = np.sum([np.sum(sum_neurons(index) > 10**-3) for index in matched_indices])
    return sum_count


def sparsity_weights(model):
    # Utility function to count number of zero weights relative to total number of weights
    layer_names_list = [layr.name for layr in model.layers]
    layers_dense = [s for s in layer_names_list if 'dense' in s]
    matched_indices = [i for i, item in enumerate(layer_names_list) if item in layers_dense]
    number_all_weights = len([model.layers[index].get_weights()[0].ravel() for index in matched_indices])
    values_weights = np.sum([np.abs(model.layers[index].get_weights()[0].ravel()) > 10**-3 for index in
                             matched_indices])
    return values_weights/number_all_weights


def test_sparsity_neurons():
    w = np.ones((256, 128))
    w[100, :] = 0
    w[50, :] = 0
    sum_neurons = np.sum(np.abs(w), axis=1)
    sum_count = np.sum([np.sum(sum_neurons > 10**-3)])
    return sum_count

def test_sparsity_weights():
    w = np.ones((256, 128))
    w[100, :] = 0
    w[50, :] = 0
    number_all_weights = len(w.ravel())
    values_weights = np.sum([np.abs(w.ravel()) > 10**-3])
    return values_weights/number_all_weights