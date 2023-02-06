import numpy as np
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


class TrainRegParameter:
    def __init__(self, alpha):
        self.alpha = alpha

    def set_alpha(self, alpha):
        K.set_value(self.alpha, alpha)


class L2(Regularizer, TrainRegParameter):
    """ Regularizer for combined L21 group regularization and L1 regularization. """

    def __init__(self, alpha=0.):
        super().__init__(alpha)
        self.alpha = alpha

    def __call__(self, x):
        regularization = 0.
        regularization += K.sum(self.alpha * K.square(x))
        return regularization

    def get_config(self):
        return {'alpha_l2': float(K.get_value(self.alpha))}


class L1(Regularizer, TrainRegParameter):
    def __init__(self, alpha):
        super().__init__(alpha)
        self.alpha = alpha

    def __call__(self, x):
        regularization = 0.
        regularization += K.sum(self.alpha * K.abs(x))
        return regularization

    def get_config(self):
        return {'alpha_l1': float(K.get_value(self.alpha))}


class L21_and_L1(Regularizer, TrainRegParameter):
    """ Regularizer for combined L21 group regularization and L1 regularization. """

    def __init__(self, alpha):
        super().__init__(alpha)
        self.alpha = alpha

    def __call__(self, x):
        regularization = 0.
        regularization += K.sum(self.alpha * K.abs(x))
        regularization += K.sum(self.alpha * K.sqrt(K.sum(K.square(x), axis=1)))
        return regularization

    def get_config(self):
        return {'alpha_l21_l1': float(K.get_value(self.alpha))}


class L21(Regularizer, TrainRegParameter):
    """ Regularizer for L21 regularization. """

    def __init__(self, alpha):
        super().__init__(alpha)
        self.alpha = alpha

    def __call__(self, x):
        # We do not add the normalization coefficient for now
        # const_coeff = np.sqrt(K.int_shape(x)[1])

        regularization = 0.
        regularization += K.sum(self.alpha * K.sqrt(K.sum(K.square(x), axis=1)))
        return regularization

    def get_config(self):
        return {'alpha_l21': float(K.get_value(self.alpha))}


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