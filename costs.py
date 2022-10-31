""" functions used to compute loss """

import numpy as np


def compute_loss_linear_regression(y, tx, w):

    """Calculate loss using MSE.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    err = y - tx.dot(w)
    return np.sum(err ** 2) / (2 * len(y))


def compute_logloss_logistic_regression(y, tx, w):

    """Calculate log-loss for logistic regression and regularized logistic regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = len(y)
    loss = np.sum(-tx.dot(w) * y + np.log(1 + np.exp(tx.dot(w))))
    return loss / N
