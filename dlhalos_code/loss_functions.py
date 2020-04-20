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


def conditional_loss(y_true, y_predicted):
    # the function K.less(x, y) returns True when x < y

    negative_term = K.square(y_true - y_predicted)
    mask_neg = K.less(y_predicted, -1 * K.ones_like(y_predicted)) # this means that is is 1 when predictions < -1

    range_term = K.square(y_true - y_predicted)
    mask_range = K.less_equal(K.abs(y_predicted), -1 * K.ones_like(y_predicted))

    positive_term = K.pow(y_true - y_predicted, 4)
    mask_pos = K.less(1 * K.ones_like(y_predicted), y_predicted)

    loss = mask_neg * negative_term + mask_range * range_term + positive_term * mask_pos
    return K.mean(loss, axis=-1)

# def cauchy_selection_boundaries():
#     term1 = K.exp(K.exp(-x))
#     term2 = K.pow(K.exp(K.e), 2)
#
#     Exp[Exp[-x]] + (-(E ^ (1 + E) / 2) + (1 + d) / ((1 + d) ^ 2 + \[Gamma] ^ 2) -
#                                                                                 2 / ((4 \[Gamma] + \[Gamma] ^ 3) ArcTan[
#         2 /\[Gamma]])) x ^ 2 +
#     1 / 2(-2 + E)
#     E ^ E - (1 + d) / ((1 + d) ^ 2 + \[Gamma] ^ 2) +
#     2 / ((4 \[Gamma] + \[Gamma] ^ 3) ArcTan[2 /\[Gamma]]) +
#     Log[1 + (1 + d) ^ 2 /\[Gamma] ^ 2] + Log[\[Gamma]] +
#     Log[ArcTan[2 /\[Gamma]]];


def cauchy_selection_loss_with_gamma(layer, y_max=1, y_min=-1, epsilon=0):
    def loss(y_true, y_pred):
        gamma = layer.gamma
        r = (y_true - y_pred)/gamma

        tail_term = K.log(1 + K.square(r))
        selection_term = K.log(K.atan((y_max - y_pred)/gamma) - K.atan((y_min - y_pred)/gamma) + epsilon)

        loss = tail_term + selection_term
        return K.mean(loss, axis=-1)


def cauchy_selection_loss(y_true, y_predicted, gamma=0.2, y_max=1, y_min=-1, epsilon=0):
    y_max = 1
    y_min = -1
    gamma = 0.2
    epsilon = 0
    r = (y_true - y_predicted)/gamma

    tail_term = K.log(1 + K.square(r))
    selection_term = K.log(K.atan((y_max - y_predicted)/gamma) - K.atan((y_min - y_predicted)/gamma) + epsilon)

    loss = tail_term + selection_term
    return K.mean(loss, axis=-1)

# def cauchy_selection_loss(gamma=0.2, y_max=1, y_min=-1, epsilon=0):
#     def loss(y_true, y_predicted):
#         r = (y_true - y_predicted) / gamma
#
#         tail_term = K.log(1 + K.square(r))
#         selection_term = K.log(K.atan((y_max - y_predicted) / gamma) - K.atan((y_min - y_predicted) / gamma) + epsilon)
#
#         loss = tail_term + selection_term
#         return K.mean(loss, axis=-1)


def cauchy_selection_loss_numpy(y_true, y_predicted):
    y_max = 1
    y_min = -1
    gamma = 1
    r = (y_true - y_predicted)/gamma
    epsilon = 10**-6

    tail_term = np.log(1 + np.square(r))
    selection_term = np.log(np.arctan((y_max - y_predicted)/gamma) - np.arctan((y_min - y_predicted)/gamma) + epsilon)

    loss = tail_term + selection_term
    return np.mean(loss, axis=-1)


def sivia_skilling_loss_numpy(y_true, y_predicted):
    epsilon = 10 ** -6
    gamma = 1
    r = abs(y_true - y_predicted)/gamma
    factor = - np.log((1 - np.exp(-r**2 / 2)) / r ** 2)
    return factor + np.log(gamma * np.sqrt(2*np.pi))

def squared_error_numpy(y_true, y_predicted):
    return (y_true - y_predicted)**2
