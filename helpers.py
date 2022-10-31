# -*- coding: utf-8 -*-
"""Some helper functions."""

import numpy as np
import csv
from preprocessing import *
from implementations import *
from crossvalidation import *


def load_train_data(path, default_missing_value=-999):
    """
    Load data function. 
    Args:
        path : path to find the file

    Returns
        ids : array of size (N,) containing ids of the observed data
        tx  : array of size (N,D) corresponding to feature matrix
        y   : prediction converted according to the rule {'b': 0, 's': 1}
    """
    columns_labels = np.genfromtxt(
        path, delimiter=",", max_rows=1, dtype=str, usecols=list(range(2, 32))
    )
    ids = np.genfromtxt(path, delimiter=",", usecols=[0], dtype=int, skip_header=1)
    tx = np.genfromtxt(path, skip_header=1, delimiter=",", usecols=list(range(2, 32)))
    y = np.genfromtxt(
        path,
        skip_header=1,
        delimiter=",",
        usecols=1,
        converters={1: lambda x: 1 if x == b"s" else 0},
        dtype=int,
    )
    tx[tx == default_missing_value] = np.nan
    return y, tx, ids, columns_labels


def load_test_data(path, missing_values=-999.0):
    """
    Load data function. 
    Args:
        path : path to find the file

    Returns
        ids : array of size (N,) containing ids of the observed data
        tx :  array if size (N,D) corresponding to feature matrix
    """
    columns_labels = np.genfromtxt(
        path, delimiter=",", max_rows=1, dtype=str, usecols=list(range(2, 32))
    )
    ids = np.genfromtxt(path, delimiter=",", usecols=[0], dtype=int, skip_header=1)
    tx = np.genfromtxt(path, skip_header=1, delimiter=",", usecols=list(range(2, 32)))
    tx[tx == missing_values] = np.nan
    return tx, ids, columns_labels


def sigmoid(x):
    """
    Compute sigmoid function for logistic regression.
    """
    return 1.0 / (1 + np.exp(-x))


def predict_logistic(tx, w, threshold):
    """
    Compute predictions for logistic regresson model.
    """
    prediction = sigmoid(tx.dot(w))
    prediction[prediction >= threshold] = 1
    prediction[prediction < threshold] = -1
    return prediction.astype(int)


def predict_ridge(tx, w, threshold=0.5):
    """ 
    Compute prediction for ridge regression. It also works for every other linear model used for classification
    """
    prediction = tx.dot(w)
    prediction[prediction >= threshold] = 1
    prediction[prediction < threshold] = -1
    return prediction.astype(int)


def divide_dataset(tx, y, ratio, seed):
    """
    Divide the dataset into train and test sets.
    
    Args
        ratio: scalar, ratio of training data w.r.t the total number of observations
    
    Returns:
        tx_train,tx_test,y_train,y_test: train and test data to use
    """
    np.random.seed(seed)
    N = int(np.floor(ratio * len(y)))
    indices = np.random.permutation(len(y))
    tx_train = tx[indices[:N]]
    tx_test = tx[indices[N:]]
    y_train = y[indices[:N]]
    y_test = y[indices[N:]]

    return tx_train, tx_test, y_train, y_test


def compute_accuracy(
    y_s,
    tx_s,
    divide_ratio,
    lambdas,
    degrees,
    no_interaction_factors,
    pred_threshold=0.5,
    method="linear",
    gamma=0.1,
):
    """ 
    Helper function to compute test and train accuracy of the class model
    """
    seeds = list(range(2, 11))
    test_accuracy = np.zeros(len(seeds))
    train_accuracy = np.zeros(len(seeds))

    for seed_indices, seed in enumerate(seeds):
        den_train, den_test = 0, 0
        for idx in range(len(tx_s)):

            x_train, x_test, y_train, y_test = divide_dataset(
                tx_s[idx], y_s[idx], divide_ratio, seed
            )

            den_train += len(y_train)
            den_test += len(y_test)

            # We compute polynomial expansion, add offset and interaction factors

            phi_train = build_poly(x_train, degrees[idx], no_interaction_factors[idx])
            phi_test = build_poly(x_test, degrees[idx], no_interaction_factors[idx])

            if method == "logistic":

                w_opt, _ = reg_logistic_regression(
                    y_train,
                    phi_train,
                    lambdas[idx],
                    np.zeros(phi_train.shape[1]),
                    300,
                    gamma,
                )
                pred_train = sigmoid(phi_train.dot(w_opt))
                pred_test = sigmoid(phi_test.dot(w_opt))
                pred_train[pred_train >= pred_threshold] = 1
                pred_train[pred_train < pred_threshold] = 0
                pred_test[pred_test >= pred_threshold] = 1
                pred_test[pred_test < pred_threshold] = 0

            if method == "linear":

                w_opt, _ = ridge_regression(y_train, phi_train, lambdas[idx])
                pred_train = phi_train.dot(w_opt)
                pred_train[pred_train >= pred_threshold] = 1
                pred_train[pred_train < pred_threshold] = 0
                pred_test = phi_test.dot(w_opt)
                pred_test[pred_test >= pred_threshold] = 1
                pred_test[pred_test < pred_threshold] = 0

            if method == "ls_normal_equations":

                w_opt, _ = least_squares(y_train, phi_train)
                pred_train = phi_train.dot(w_opt)
                pred_train[pred_train >= pred_threshold] = 1
                pred_train[pred_train < pred_threshold] = 0
                pred_test = phi_test.dot(w_opt)
                pred_test[pred_test >= pred_threshold] = 1
                pred_test[pred_test < pred_threshold] = 0

            train_accuracy[seed_indices] += np.sum(pred_train == y_train)
            test_accuracy[seed_indices] += np.sum(pred_test == y_test)

        train_accuracy[seed_indices] /= den_train
        test_accuracy[seed_indices] /= den_test

    print("Average train accuracy: {}".format(np.mean(train_accuracy)))
    print("std train accuracy: {}".format(np.std(train_accuracy)))
    print("Average test accuracy: {}".format(np.mean(test_accuracy)))
    print("std train accuracy: {}".format(np.std(test_accuracy)))


def create_submission(ids, y_pred, name, file_name):
    """
    Create a csv file to submit the output to the challenge arena.
    """
    with open(file_name, "w", newline="") as file:
        dw = csv.DictWriter(file, delimiter=",", fieldnames=name)
        dw.writeheader()
        for r1, r2 in zip(ids, y_pred):
            dw.writerow({"Id": r1, "Prediction": r2})
