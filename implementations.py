"Useful functions to use during the project "

import numpy as np
from costs import *
from gradients import *


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):

    """The Gradient Descent (GD) algorithm for linear regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) for the final iteration of the method
        w: the model parameters as numpy arrays of shape (D, )
        """
    w = initial_w
    for n in range(max_iters):
        grad = compute_gradient_linear_regression(y, tx, w)
        w = w - grad * gamma
    loss = compute_loss_linear_regression(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):

    """The Stochastic Gradient Descent algorithm (SGD) for linear regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) for the last iteration of SGD
        w: the model parameters as numpy arrays of shape (D, ), for the final iteration of SGD
    """
    w = initial_w
    for n in range(max_iters):
        for y_minibatch, tx_minibatch in batch_iter(y, tx):
            stoch_grad = compute_stoch_gradient(y_minibatch, tx_minibatch, w)
        w = w - gamma * stoch_grad
    loss = compute_loss_linear_regression(y, tx, w)
    return w, loss


def least_squares(y, tx):

    """ The least squares algorithm for linear regression using normal equations.
    Args:
        y : shape = (N,)
        tx : shape = (N,D)
    Returns:
        w : the optimal model parameters as numpy arrays of shape (D,)
        loss: the loss value (scalar) for least squares"""

    A = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(A, b)
    return w, compute_loss_linear_regression(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression using normal equations.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        ridge_loss: the loss value (scalar) for ridge regression.
    """
    lambda_tilde = 2 * lambda_ * len(y)
    A = tx.T.dot(tx) + lambda_tilde * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    ridge_loss = compute_loss_linear_regression(y, tx, w)
    return w, ridge_loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):

    """The Gradient Descent (GD) algorithm for logistic regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) for the final iteration of the method
        w: the model parameters as numpy arrays of shape (D, )
        """
    w = initial_w
    for n in range(max_iters):
        grad = compute_gradient_logistic_regression(y, tx, w)
        w = w - gamma * grad
    loss = compute_logloss_logistic_regression(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    """The Gradient Descent (GD) algorithm for regularized logistic regression.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        loss: the loss value (scalar) for the final iteration of the method
        w: the model parameters as numpy arrays of shape (D, )
        """
    w = initial_w
    N = len(y)
    for n in range(max_iters):
        grad = compute_gradient_logistic_regression(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = compute_logloss_logistic_regression(y, tx, w)
    return w, loss
