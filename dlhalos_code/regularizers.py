import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K


def l2_norm(alpha):
    return regularizers.l2(alpha)


def l1_norm(alpha):
    return regularizers.l1(alpha)


def l1_and_l21_group(alpha):
    return l1_norm(alpha) + l21_group(alpha)


def l21_group(alpha):
    return L21(alpha)


class L21(Regularizer):
    """ Regularizer for L21 regularization. """

    def __init__(self, alpha=0.):
        self.alpha = K.cast_to_floatx(alpha)

    def __call__(self, x):
        # We do not add the normalization coefficient for now
        # const_coeff = np.sqrt(K.int_shape(x)[1])

        reg = self.alpha * K.sum(K.sqrt(K.sum(K.square(x), axis=1)))
        return reg


# Utility function to count active neurons in a Keras model with Dense layers
def count_neurons(model):
    return np.sum([np.sum(np.sum(np.abs(l.get_weights()[0]), axis=1) > 10**-3) \
                          for l in model.layers])