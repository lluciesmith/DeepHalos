import tensorflow.keras.backend as K
import tensorflow.math as math
import numpy as np


def sivia_skilling_loss(y_true, y_predicted):
    epsilon = 10 ** -6
    gamma = 0.2
    r = abs(y_true - y_predicted)/gamma
    factor = - K.log((1 - K.exp(-(r ** 2 + epsilon) / 2)) / (r ** 2 + epsilon))
    return K.mean(factor, axis=-1)


def cauchy_selection_loss(y_true, y_predicted):
    y_max = 1
    y_min = -1
    gamma = 0.2
    epsilon = 10**-6
    r = (y_true - y_predicted)/gamma

    tail_term = K.log(1 + K.square(r))
    selection_term = K.log(math.atan((y_max - y_predicted)/gamma) - math.atan((y_min - y_predicted)/gamma) + epsilon)

    loss = tail_term + selection_term
    return K.mean(loss, axis=-1)


def cauchy_selection_loss_numpy(y_true, y_predicted):
    y_max = 1
    y_min = -1
    gamma = 1
    r = (y_true - y_predicted)/gamma

    tail_term = np.log(1 + np.square(r))
    selection_term = np.log(np.arctan((y_max - y_predicted)/gamma) - np.arctan((y_min - y_predicted)/gamma))

    loss = tail_term + selection_term
    return loss


def sivia_skilling_loss_numpy(y_true, y_predicted):
    epsilon = 10 ** -6
    gamma = 1
    r = abs(y_true - y_predicted)/gamma
    factor = - np.log((1 - np.exp(-(r ** 2 + epsilon) / 2)) / (r ** 2 + epsilon))
    return factor
