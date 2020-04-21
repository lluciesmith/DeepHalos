import tensorflow.keras.backend as K
import tensorflow.math as math
import tensorflow as tf
import numpy as np


def sivia_skilling_loss(y_true, y_predicted):
    epsilon = 10 ** -6
    gamma = 0.2
    r = abs(y_true - y_predicted)/gamma
    factor = - K.log((1 - K.exp(-(r ** 2 + epsilon) / 2)) / (r ** 2 + epsilon))
    return K.mean(factor, axis=-1)


def cauchy_selection_loss_fixed_boundary(gamma=0.2, y_max=1, y_min=-1):
    L = ConditionalCauchySelectionLoss(gamma=gamma, y_max=y_max, y_min=y_min)

    def loss(y_true, y_predicted):
        return L.loss(y_true, y_predicted)
    return loss


def cauchy_selection_loss_trainable_gamma(layer, y_max=1, y_min=-1):
    def loss(y_true, y_pred):
        gamma = layer.gamma
        r = (y_true - y_pred)/gamma
        tail_term = K.log(1 + K.square(r))
        selection_term = K.log(tf.atan((y_max - y_pred)/gamma) - tf.atan((y_min - y_pred)/gamma))
        loss = tail_term + selection_term + K.log(gamma)
        return K.mean(loss, axis=-1)
    return loss


def cauchy_selection_loss(gamma=0.2, y_max=1, y_min=-1):
    def loss(y_true, y_predicted):
        r = (y_true - y_predicted) / gamma
        tail_term = K.log(1 + K.square(r))
        selection_term = K.log(tf.atan((y_max - y_predicted) / gamma) - tf.atan((y_min - y_predicted) / gamma))
        loss = tail_term + selection_term
        return K.mean(loss, axis=-1)
    return loss


# Numpy equivalents

def cauchy_selection_loss_numpy(y_true, y_predicted, y_max=1, y_min=-1, gamma=1):
    r = (y_true - y_predicted)/gamma

    tail_term = np.log(1 + np.square(r))
    selection_term = np.log(np.arctan((y_max - y_predicted)/gamma) - np.arctan((y_min - y_predicted)/gamma))

    loss = tail_term + selection_term
    return loss


def sivia_skilling_loss_numpy(y_true, y_predicted, gamma=1):
    r = abs(y_true - y_predicted)/gamma
    factor = - np.log((1 - np.exp(-r**2 / 2)) / r ** 2)
    norm = np.log(gamma * np.sqrt(2*np.pi))
    return factor + norm


def squared_error_numpy(y_true, y_predicted):
    return (y_true - y_predicted)**2


# Class for loss Cauchy + selection loss function with fixed boundary


class ConditionalCauchySelectionLoss:
    def __init__(self, gamma=0.2, y_max=1, y_min=-1):
        self.g = gamma
        self.y_maximum = y_max
        self.y_minimum = y_min
        # e = K.constant(np.e, dtype="float32")
        self.e = np.e

    def loss(self, y_true, y_pred):
        return self._loss(y_true, y_pred)

    def _loss(self, y_true, y_pred):
        zeros = K.zeros_like(y_pred)

        mask_range = K.less(K.abs(y_pred), K.ones_like(y_pred))
        range_term = tf.where(mask_range, self.loss_range(y_true, y_pred), zeros)

        mask_neg = K.less_equal(y_pred, -1 * K.ones_like(y_pred))
        negative_term = tf.where(mask_neg, self.loss_neg(y_true, y_pred), zeros)

        mask_pos = K.less_equal(K.ones_like(y_pred), y_pred)
        positive_term = tf.where(mask_pos, self.loss_pos(y_true, y_pred), zeros)

        loss = negative_term + range_term + positive_term
        return K.mean(loss, axis=-1)

    def loss_pos(self, y_true, y_pred):
        g, y_maximum, e = self.g, self.y_maximum, self.e
        alpha_pos = self.alpha(y_maximum, y_true, e, g)
        beta_pos = self.beta(y_maximum, y_true, e, g)
        return self.function_outside_boundary(y_pred, y_maximum, alpha_pos, beta_pos)

    def loss_neg(self, y_true, y_pred):
        g, y_minimum, e = self.g, self.y_minimum, self.e
        alpha_neg = self.alpha(y_minimum, y_true, e, g)
        beta_neg = self.beta(y_minimum, y_true, e, g)
        return self.function_outside_boundary(y_pred, y_minimum, alpha_neg, beta_neg)

    def loss_range(self, y_true, y_pred):
        return self.function_inside_boundary(y_true, y_pred, self.g, self.y_maximum, self.y_minimum)

    def function_outside_boundary(self, y_pred, y_boundary, alpha, beta):
        return K.exp(K.exp(y_boundary * y_pred)) + alpha * K.square(y_pred) + beta

    def function_inside_boundary(self, y_true, y_pred, gamma, y_max, y_min):
        r = (y_true - y_pred) / gamma
        f = K.log(gamma) + K.log(1 + K.square(r)) + K.log(
            K.atan((y_max - y_pred) / gamma) - K.atan((y_min - y_pred) / gamma))
        return f

    def alpha(self, y_boundary, y_true, e_const, gamma):
        term1 = - K.exp(1 + e_const) / 2
        term2 = - y_boundary * (y_true - y_boundary) / (K.pow(-y_boundary + y_true, 2) + K.pow(gamma, 2))
        term3 = - 2 / ((4 * gamma + K.pow(gamma, 3)) * K.atan(2 / gamma))
        return term1 + term2 + term3

    def beta(self, y_boundary, y_true, e_const, gamma):
        term1 = 0.5 * (-2 + e_const) * K.exp(e_const)
        term2 = y_boundary * (y_true - y_boundary) / (K.pow(-y_boundary + y_true, 2) + K.pow(gamma, 2))
        term3 = 2 / ((4 * gamma + K.pow(gamma, 3)) * K.atan(2 / gamma))
        term4 = K.log(1 + (K.pow(- y_boundary + y_true, 2) / K.pow(gamma, 2)))
        term5 = K.log(gamma)
        term6 = K.log(K.atan(2 / gamma))
        return term1 + term2 + term3 + term4 + term5 + term6

