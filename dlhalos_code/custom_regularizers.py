import numpy as np
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K


def l2_norm(alpha):
    return L2(alpha)


def l1_norm(alpha, layer=None):
    return L1(alpha, layer=layer)


def l1_and_l21_group(alpha, layer=None):
    return L21_and_L1(alpha, layer=layer)


def l21_group(alpha, layer=None):
    return L21(alpha, layer=layer)


class RegClass:
    def __init__(self, alpha, layer=None):
        self.layer = layer
        if self.layer is None:
            self.alpha = self.alpha
        if self.layer is not None:
            self.alpha = layer.alpha

    # def set_alpha_from_layer(self, layer):
    #     self.alpha = K.get_value(layer.gamma)
    #     # val_alpha = K.get_value(alpha)[0]
    #     # variable_alpha = K.variable(self.alpha)
    #     # K.set_value(variable_alpha, val_alpha)


class L2(Regularizer):
    """ Regularizer for combined L21 group regularization and L1 regularization. """

    def __init__(self, alpha=0.):
        # super().__init__(alpha)
        self.alpha = alpha

    def __call__(self, x):
        # if self.layer is not None:
        #     self.set_alpha_from_layer(self.layer)

        regularization = 0.
        regularization += self.alpha * K.sum(K.square(x))
        return regularization

    def get_config(self):
        return {'alpha_l2': float(K.get_value(self.alpha))}


class L1(RegClass, Regularizer):
    """ Regularizer for combined L21 group regularization and L1 regularization. """

    # def __init__(self, alpha=0., layer=None):
    #     super().__init__(alpha)
    #     self.layer = layer
    #
    # def __call__(self, x):
    #     if self.layer is not None:
    #         self.set_alpha_from_layer(self.layer)
    def __init__(self, alpha):
        super().__init__(alpha)

    def __call__(self, x):
        # if self.layer is not None:
        #     self.set_alpha_from_layer(self.layer)

        regularization = 0.
        regularization += self.alpha * K.sum(K.abs(x))
        return regularization

    def get_config(self):
        return {'alpha_l1': float(K.get_value(self.alpha))}


class L21_and_L1(RegClass, Regularizer):
    """ Regularizer for combined L21 group regularization and L1 regularization. """

    def __init__(self, alpha=0., layer=None):
        super().__init__(alpha, layer=layer)

    def __call__(self, x):
        regularization = 0.
        regularization += self.alpha * K.sum(K.abs(x))
        regularization += self.alpha * K.sum(K.sqrt(K.sum(K.square(x), axis=1)))
        return regularization

    def get_config(self):
        return {'alpha_l21_l1': float(K.get_value(self.alpha))}


class L21(RegClass, Regularizer):
    """ Regularizer for L21 regularization. """

    def __init__(self, alpha=0., layer=None):
        super().__init__(alpha, layer=layer)

    def __call__(self, x):
        # We do not add the normalization coefficient for now
        # const_coeff = np.sqrt(K.int_shape(x)[1])

        regularization = 0.
        regularization += self.alpha * K.sum(K.sqrt(K.sum(K.square(x), axis=1)))
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