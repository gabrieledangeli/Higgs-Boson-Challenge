import numpy as np
from implementations import *
from costs import *
from preprocessing import *


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_log(
    y, x, k_indices, k, lambda_, gamma, degree, max_iters, no_interaction_factors
):
    """
    Return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        gamma :     scalar, the learning rate
        degree:     scalar, cf. build_poly()
        max_iters : scalar
        no_interaction_factors : number of trigonometric features in the dataset

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """

    num_row = y.shape[0]

    train_indices = np.concatenate(
        [k_indices[:k, :].ravel(), k_indices[k + 1 :, :].ravel()]
    )
    test_indices = k_indices[k]

    x_train = x[train_indices]
    x_test = x[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    # We compute polynomial expansion (and automatically add the offset column)

    poly_train = build_poly(x_train, degree, no_interaction_factors)
    poly_test = build_poly(x_test, degree, no_interaction_factors)

    # Finding optimal weights

    initial_w = np.zeros(poly_train.shape[1])

    w_opt, _ = reg_logistic_regression(
        y_train, poly_train, lambda_, initial_w, max_iters, gamma
    )

    loss_tr = compute_logloss_logistic_regression(y_train, poly_train, w_opt)
    loss_te = compute_logloss_logistic_regression(y_test, poly_test, w_opt)

    return loss_tr, loss_te


def cross_validation_demo_log(
    y, tx, k_fold, lambdas, gamma, max_iters, degrees, no_interaction_factors, seed=10
):
    """
    Cross validation over regularisation parameter lambda.
    
    Args:
        degrees: list of degrees of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
        gamma :     scalar, the learning rate
        degrees: =  shape (q, ) where q is the number of degrees to test
        max_iters : scalar
        no_interaction_factors : number of trigonometric features in the dataset
    Returns:
        best degree : a scalar, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda and best degree
    """

    k_fold = k_fold
    lambdas = lambdas
    # Split data in k fold
    k_idx = build_k_indices(y, k_fold, seed)

    # Define matrices to store the loss of training data and test data

    rmse_tr = np.zeros((len(lambdas), len(degrees)))
    rmse_te = np.zeros((len(lambdas), len(degrees)))

    for i, param in enumerate(lambdas):
        for j, deg in enumerate(degrees):
            l_tr = 0
            l_te = 0
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation_log(
                    y,
                    tx,
                    k_idx,
                    k,
                    param,
                    gamma,
                    deg,
                    max_iters,
                    no_interaction_factors,
                )
                l_te += loss_te
                l_tr += loss_tr
            l_te = l_te / k_fold
            l_tr = l_tr / k_fold
            rmse_tr[i, j] = l_tr
            rmse_te[i, j] = l_te

    idx_lambda, idx_degree = np.unravel_index(np.argmin(rmse_te), rmse_te.shape)
    best_degree, best_lambda, best_rmse = (
        degrees[idx_degree],
        lambdas[idx_lambda],
        rmse_te[idx_lambda, idx_degree],
    )

    print(
        "The choice of lambda which leads to the best test logloss is %.5f with a test logloss of %.3f. The best degree is %.1f"
        % (best_lambda, best_rmse, best_degree)
    )
    return best_degree, best_lambda, best_rmse


def cross_validation_ridge(y, x, k_indices, k, lambda_, degree, no_interaction_factors):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()
        no_interaction_factors : number of trigonometric features in the dataset
    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """

    num_row = y.shape[0]

    train_indices = np.concatenate(
        [k_indices[:k, :].ravel(), k_indices[k + 1 :, :].ravel()]
    )
    test_indices = k_indices[k]

    x_train = x[train_indices]
    x_test = x[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    # We compute polynomial expansion (and automatically add the offset column)

    poly_train = build_poly(x_train, degree, no_interaction_factors)
    poly_test = build_poly(x_test, degree, no_interaction_factors)

    # Finding optimal weights

    w_opt, _ = ridge_regression(y_train, poly_train, lambda_)

    mse_train = compute_loss_linear_regression(y_train, poly_train, w_opt)
    mse_test = compute_loss_linear_regression(y_test, poly_test, w_opt)
    loss_tr = np.sqrt(2 * mse_train)
    loss_te = np.sqrt(2 * mse_test)

    return loss_tr, loss_te


def cross_validation_demo_ridge(
    y, tx, k_fold, lambdas, degrees, no_interaction_factors, seed=12
):
    """cross validation over regularisation parameter lambda and hyperparameter degree.
    
    Args:
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
        degrees: list of degrees for polynomial expansion
        no_interaction_factors : number of trigonometric features in the dataset
    Returns:
        best_degree : scalar, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best pair (lambda,degree)
    """
    k_fold = k_fold
    lambdas = lambdas
    # Split data in k fold
    k_idx = build_k_indices(y, k_fold, seed)

    # Define matrices to store loss of test data

    rmse_tr = np.zeros((len(lambdas), len(degrees)))
    rmse_te = np.zeros((len(lambdas), len(degrees)))

    for i, param in enumerate(lambdas):
        for j, deg in enumerate(degrees):
            l_tr = 0
            l_te = 0
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation_ridge(
                    y, tx, k_idx, k, param, deg, no_interaction_factors
                )
                l_te += loss_te
                l_tr += loss_tr
            l_te = l_te / k_fold
            l_tr = l_tr / k_fold
            rmse_tr[i, j] = l_tr
            rmse_te[i, j] = l_te

    idx_lambda, idx_degree = np.unravel_index(np.argmin(rmse_te), rmse_te.shape)
    best_degree, best_lambda, best_rmse = (
        degrees[idx_degree],
        lambdas[idx_lambda],
        rmse_te[idx_lambda, idx_degree],
    )

    print(
        "The choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f. The best degree is %.1f"
        % (best_lambda, best_rmse, best_degree)
    )

    return best_degree, best_lambda, best_rmse
