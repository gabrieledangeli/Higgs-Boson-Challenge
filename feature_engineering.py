""" File to compute preprocessing routine """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from helpers import *
from implementations import *
from crossvalidation import *
from preprocessing import *
from dataset_splitting import *


def preprocessing(tx, y, ids, features):
    """
    Function to preprocess data, managing missing values, capping outliers, removing useless features
    and adding trigonometric functions
    
    Args:
        data:  array of shape (N,D) corresponding to feature matrix
        y :    array of size (N,) containing outcomes
        ids :  array of size (N, ) containing id for each row in tx
        features:  array of size(D,) containing names for each feature
    
    Returns:
        list_subsets:   list containing feature matrices for each subset obtained according to categorical variable PRI_jet_num
        list_features:  list containing names of useful features for each subset
        y_i :           outcomes for each subset
        columns_to_drop_in_subset : list containing the indices of columns to remove in test dataset for each subset
    """

    # Retrieving logical masks to divide the dataset

    categorical_column = np.where(features == "PRI_jet_num")[0][0]
    mask_0, mask_1, mask_2_3 = divide_indices_in_subsets(tx, categorical_column)

    # Removing categorical column since it is now useless

    tx = np.delete(tx, categorical_column, axis=1)
    # since we delete the column in tx, we also delete the name of the categorical feature used to divide the dataset
    features = np.delete(features, categorical_column)

    # Splitting train dataset, the output vector and ids w.r.t according to the mask

    subset_0, y_0, ids_0 = divide_train_dataset_in_subsets(tx, y, ids, mask_0)
    subset_1, y_1, ids_1 = divide_train_dataset_in_subsets(tx, y, ids, mask_1)
    subset_2_3, y_2_3, ids_2_3 = divide_train_dataset_in_subsets(tx, y, ids, mask_2_3)

    # Defining a list containing each subset

    list_subsets = [subset_0, subset_1, subset_2_3]

    # Define a list containing features for each subset

    list_features = [features] * 3

    # Managing missing values in each subset of data

    columns_to_drop_in_subsets = [0] * 3
    for idx in range(3):
        (
            list_subsets[idx],
            list_features[idx],
            columns_to_drop_in_subsets[idx],
        ) = managing_missing_values(list_subsets[idx], features)

    # The last column in subset_0 is a zeros vector (see the documentation). Therefore, we drop it not to have problems when standardizing

    list_subsets[0] = np.delete(list_subsets[0], -1, 1)
    list_features[0] = np.delete(list_features[0], -1)

    # Defining trigonometric features (sine and cosine) starting from columns containing values

    columns_angles_0 = [11, 14, 16]
    columns_angles_1 = [11, 14, 16, 20]
    columns_angles_2 = [15, 18, 20, 27]

    list_subsets[0], list_features[0] = trigonometrics(
        list_subsets[0], columns_angles_0, list_features[0]
    )
    list_subsets[1], list_features[1] = trigonometrics(
        list_subsets[1], columns_angles_1, list_features[1]
    )
    list_subsets[2], list_features[2] = trigonometrics(
        list_subsets[2], columns_angles_2, list_features[2]
    )

    # Applying logarithmic transformation to skewed distributions in each subset ( X_new = log(1+x_old) )

    to_log_c0 = [0, 1, 2, 3, 5, 6, 7, 9, 11, 13, 14]
    to_log_c1 = [0, 1, 2, 3, 5, 6, 7, 9, 11, 13, 14, 15, 17]
    to_log_c2 = [0, 1, 2, 3, 5, 8, 9, 10, 13, 15, 17, 18, 19, 22, 24]

    list_subsets[0][:, to_log_c0] = log_transform(list_subsets[0][:, to_log_c0])
    list_subsets[1][:, to_log_c1] = log_transform(list_subsets[1][:, to_log_c1])
    list_subsets[2][:, to_log_c2] = log_transform(list_subsets[2][:, to_log_c2])

    # Handling outliers by replacing them with 5% or 95% percentiles

    for idx in range(3):
        list_subsets[idx] = capping_outliers(list_subsets[idx])

    # Identifying useless variables and dropping corresponding columns in each subset

    useless_c0 = [3, 5, 8, 16, 17, 19, 20]
    useless_c1 = [11, 19, 20, 21, 22, 23, 24, 25]
    useless_c2 = [15, 17, 18, 19, 21, 22, 26, 27, 28, 30, 31, 32]

    a = list(range(list_subsets[0].shape[1]))
    useful_c0 = np.delete(a, useless_c0)
    list_subsets[0] = list_subsets[0][:, useful_c0]
    list_features[0] = list_features[0][useful_c0]

    b = list(range(list_subsets[1].shape[1]))
    useful_c1 = np.delete(b, useless_c1)
    list_subsets[1] = list_subsets[1][:, useful_c1]
    list_features[1] = list_features[1][useful_c1]

    c = list(range(list_subsets[2].shape[1]))
    useful_c2 = np.delete(c, useless_c2)
    list_subsets[2] = list_subsets[2][:, useful_c2]
    list_features[2] = list_features[2][useful_c2]

    return list_subsets, list_features, y_0, y_1, y_2_3, columns_to_drop_in_subsets
