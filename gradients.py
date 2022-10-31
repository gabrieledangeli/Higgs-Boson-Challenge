""" functions used to compute gradients """

import numpy as np
from costs import *


def compute_gradient_linear_regression(y, tx, w):

    """Compute the gradient at w for linear regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    return -tx.T.dot(err) / len(y)


def compute_stoch_gradient(y, tx, w):

    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
       This implementation holds whenever the number of batches is equal to 1.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        An array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    err = y - tx.dot(w)
    B = len(y)
    return -tx.T.dot(err) / B


def compute_gradient_logistic_regression(y, tx, w):

    """Compute the gradient at w for logistic regression (gradient of -log likelihood 
       we want to minimize).
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = len(y)
    sigma = 1.0 / (1 + np.exp(-tx.dot(w)))
    grad = tx.T.dot(sigma - y)
    return grad / N
