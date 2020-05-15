import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K


def l2_norm(alpha):
    alpha = K.cast_to_floatx(alpha)
    return regularizers.l2(alpha)


def l1_norm(alpha):
    alpha = K.cast_to_floatx(alpha)
    return regularizers.l1(alpha)


def l1_and_l21_group(alpha):
    return L21_and_L1(alpha)


def l21_group(alpha):
    return L21(alpha)


class L21_and_L1(Regularizer):
    """ Regularizer for combined L21 group regularization and L1 regularization. """

    def __init__(self, alpha=0.):
        self.alpha = K.cast_to_floatx(alpha)

    def __call__(self, x):
        regularization = 0.
        regularization += self.alpha * K.sum(K.abs(x))
        regularization += self.alpha * K.sum(K.sqrt(K.sum(K.square(x), axis=1)))
        return regularization

    def get_config(self):
        return {'alpha': self.alpha}


class L21(Regularizer):
    """ Regularizer for L21 regularization. """

    def __init__(self, alpha=0.):
        self.alpha = K.cast_to_floatx(alpha)

    def __call__(self, x):
        # We do not add the normalization coefficient for now
        # const_coeff = np.sqrt(K.int_shape(x)[1])

        reg = self.alpha * K.sum(K.sqrt(K.sum(K.square(x), axis=1)))
        return reg

    def get_config(self):
        return {'alpha': self.alpha}


# Utility function to count active neurons in a Keras model with Dense layers
def count_neurons(model):
    return np.sum([np.sum(np.sum(np.abs(l.get_weights()[0]), axis=1) > 10**-3) \
                          for l in model.layers])