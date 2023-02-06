import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf


def reconstruction_loss(truth, predicted):
    # reconstruction_loss = tf.reduce_mean(mse(truth, predicted))
    # assert len(truth.shape) == 4
    # assert truth.shape[1] == truth.shape[2]
    # original_dim = truth.shape[1] * truth.shape[2]
    # reconstruction_loss *= original_dim
    reconstruction_loss = mean_squared_error(truth, predicted)
    return reconstruction_loss


def KL_loss(z_mu, log_z_variance):
    kl_loss = 1 + log_z_variance - tf.math.square(z_mu) - tf.math.exp(log_z_variance)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    return kl_loss


def vae_loss(truth, predicted, z_mean, z_log_var, beta=1):
    return reconstruction_loss(truth, predicted) + beta * KL_loss(z_mean, z_log_var)


def mean_squared_error(y_true, y_predicted):
    err = K.square(y_true - y_predicted)
    return K.mean(err, axis=-1)


def sivia_skilling_loss(y_true, y_predicted):
    epsilon = 10 ** -6
    gamma = 0.2
    r = abs(y_true - y_predicted)/gamma
    factor = - K.log((1 - K.exp(-(r ** 2 + epsilon) / 2)) / (r ** 2 + epsilon))
    return K.mean(factor, axis=-1)


def cauchy_selection_loss_fixed_boundary_trainable_gamma(layer, y_max=1, y_min=-1, dtype="float32"):
    L = ConditionalCauchySelectionLoss(gamma=layer.gamma, y_max=y_max, y_min=y_min, dtype=dtype)

    def loss(y_true, y_predicted):
        return L.loss(y_true, y_predicted)
    return loss


def cauchy_selection_loss_fixed_boundary(gamma=0.2, y_max=1, y_min=-1, dtype="float32"):
    L = ConditionalCauchySelectionLoss(gamma=gamma, y_max=y_max, y_min=y_min, dtype=dtype)

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

def cauchy_selection_loss_numpy(y_true, y_predicted, y_max=1, y_min=-1, gamma=0.2):
    r = (y_true - y_predicted)/gamma

    tail_term = np.log(1 + np.square(r))
    selection_term = np.log(np.arctan((y_max - y_predicted)/gamma) - np.arctan((y_min - y_predicted)/gamma))

    loss = tail_term + selection_term
    return np.mean(loss, axis=-1)


def dc_dx(y_true, y_predicted, y_max=1, y_min=-1, gamma=0.2):
    d1 = - 2 * (y_true - y_predicted)/(gamma**2 + (y_true - y_predicted)**2)

    term1 = 1/(np.arctan((y_max - y_predicted)/gamma) - np.arctan((y_min - y_predicted)/gamma))
    term2 = (- gamma / (gamma**2 + (y_max - y_predicted)**2)) - (- gamma / (gamma**2 + (y_min - y_predicted)**2))
    d2 = term1 * term2
    return np.mean(d1 + d2, axis=-1)


def sivia_skilling_loss_numpy(y_true, y_predicted, gamma=1):
    r = abs(y_true - y_predicted)/gamma
    factor = - np.log((1 - np.exp(-r**2 / 2)) / r ** 2)
    norm = np.log(gamma * np.sqrt(2*np.pi))
    return factor + norm


def squared_error_numpy(y_true, y_predicted, derivative=False):
    if derivative is True:
        return np.mean(- 2 * (y_true - y_predicted), axis=-1)
    else:
        return np.mean((y_true - y_predicted)**2, axis=-1)


class ConditionalCauchySelectionLoss:
    # Class for loss Cauchy + selection loss function with fixed boundary
    def __init__(self, gamma=0.2, y_max=1, y_min=-1, dtype="float32"):
        
        if local_machine:
            self.g = gamma
            self.y_maximum = y_max
            self.y_minimum = y_min
            self.e = np.e
        else:
            self.dtype = dtype
            
            if self.dtype == "float64":
                self.g = tf.cast(gamma, "float64")
                self.y_maximum = tf.cast(y_max, "float64")
                self.y_minimum = tf.cast(y_min, "float64")
                self.e = K.constant(np.e, dtype="float64")
            else:
                self.g = gamma
                self.y_maximum = y_max
                self.y_minimum = y_min
                self.e =  K.constant(np.e, dtype="float32")

    def loss(self, y_true, y_pred):
        return self._loss(y_true, y_pred)

    def _loss(self, y_true, y_pred):
        zeros = K.zeros_like(y_pred)

        mask_range = K.less_equal(K.abs(y_pred), K.ones_like(y_pred)) # select y_pred less than one.
        range_term = tf.where(mask_range, self.loss_range(y_true, y_pred), zeros)

        mask_neg = K.less(y_pred, -1 * K.ones_like(y_pred))
        negative_term = tf.where(mask_neg, self.loss_neg(y_true, y_pred), zeros)

        mask_pos = K.less(K.ones_like(y_pred), y_pred)
        positive_term = tf.where(mask_pos, self.loss_pos(y_true, y_pred), zeros)

        loss = negative_term + range_term + positive_term
        return K.mean(loss, axis=-1)

    def dloss(self, y_true, y_pred):
        zeros = K.zeros_like(y_pred)

        mask_range = K.less_equal(K.abs(y_pred), K.ones_like(y_pred))
        range_term = tf.where(mask_range, self.deriv_loss_range(y_true, y_pred), zeros)

        mask_neg = K.less(y_pred, -1 * K.ones_like(y_pred))
        negative_term = tf.where(mask_neg, self.deriv_loss_neg(y_true, y_pred), zeros)

        mask_pos = K.less(K.ones_like(y_pred), y_pred)
        positive_term = tf.where(mask_pos, self.deriv_loss_pos(y_true, y_pred), zeros)

        loss = negative_term + range_term + positive_term
        return K.mean(loss, axis=-1)

    def loss_pos(self, y_true, y_pred):
        g, y_maximum, e = self.g, self.y_maximum, self.e
        alpha_pos = self.alpha(y_maximum, y_true, e, g)
        beta_pos = self.beta(y_maximum, y_true, e, g)
        return self.function_outside_boundary(y_pred, y_maximum, alpha_pos, beta_pos)

    def deriv_loss_pos(self, y_true, y_pred):
        g, y_maximum, e = self.g, self.y_maximum, self.e
        alpha_pos = self.alpha(y_maximum, y_true, e, g)
        return K.exp(K.exp(y_pred) + y_pred) + 2 * alpha_pos * y_pred

    def loss_neg(self, y_true, y_pred):
        g, y_minimum, e = self.g, self.y_minimum, self.e
        alpha_neg = self.alpha(y_minimum, y_true, e, g)
        beta_neg = self.beta(y_minimum, y_true, e, g)
        return self.function_outside_boundary(y_pred, y_minimum, alpha_neg, beta_neg)

    def deriv_loss_neg(self, y_true, y_pred):
        g, y_minimum, e = self.g, self.y_minimum, self.e
        alpha_neg = self.alpha(y_minimum, y_true, e, g)
        return - K.exp(K.exp(- y_pred) - y_pred) + 2 * alpha_neg * y_pred

    def loss_range(self, y_true, y_pred):
        return self.function_inside_boundary(y_true, y_pred, self.g, self.y_maximum, self.y_minimum)

    def deriv_loss_range(self, y_true, y_pred):
        y_max = self.y_maximum
        y_min = self.y_minimum

        d1 = - 2 * (y_true - y_pred) / (self.g ** 2 + (y_true - y_pred) ** 2)

        term1 = 1 / (np.arctan((y_max - y_pred) / self.g) - np.arctan((y_min - y_pred) / self.g))
        term2 = (- self.g / (self.g ** 2 + (y_max - y_pred) ** 2)) + ( self.g / (self.g ** 2 + (y_min - y_pred) ** 2))
        d2 = term1 * term2
        return d1 + d2

    def function_outside_boundary(self, y_pred, y_boundary, alpha, beta):
        return K.exp(K.exp(y_boundary * y_pred)) + alpha * K.square(y_pred) + beta
    
    def function_inside_boundary(self, y_true, y_pred, gamma, y_max, y_min):
        r = (y_true - y_pred) / gamma
        f = K.log(gamma) + K.log(1 + K.square(r)) + \
            K.log(tf.atan((y_max - y_pred) / gamma) - tf.atan((y_min - y_pred) / gamma))
        return f

    def alpha(self, y_boundary, y_true, e_const, gamma):
        term1 = - K.exp(1 + e_const) / 2
        term2 = - y_boundary * (y_true - y_boundary) / (K.pow(-y_boundary + y_true, 2) + K.pow(gamma, 2))
        term3 = - 2 / ((4 * gamma + K.pow(gamma, 3)) * tf.atan(2 / gamma))
        return term1 + term2 + term3

    def beta(self, y_boundary, y_true, e_const, gamma):
        term1 = 0.5 * (-2 + e_const) * K.exp(e_const)
        term2 = y_boundary * (y_true - y_boundary) / (K.pow(-y_boundary + y_true, 2) + K.pow(gamma, 2))
        term3 = 2 / ((4 * gamma + K.pow(gamma, 3)) * tf.atan(2 / gamma))
        term4 = K.log(1 + (K.pow(- y_boundary + y_true, 2) / K.pow(gamma, 2)))
        term5 = K.log(gamma)
        term6 = K.log(tf.atan(2 / gamma))
        return term1 + term2 + term3 + term4 + term5 + term6