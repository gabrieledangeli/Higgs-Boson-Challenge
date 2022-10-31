"""
Some  functions for dataset division wrt the categorical variable
"""
import numpy as np


def divide_indices_in_subsets(tx, categorical_column):
    """
    Function that returns boolean mask to divide dataset in categories according to PRI_num_jet variable
    Args:
        tx : array of shape (N,D)
        categorical_column : number of column containing categorical value
    
    Returns
        indices_cat_i :  rows s.t tx[indices_cat_i,categorical_column] = i
    """

    indices_cat_0 = tx[:, categorical_column] == np.float(0)
    indices_cat_1 = tx[:, categorical_column] == np.float(1)
    indices_cat_2_3 = (tx[:, categorical_column] == np.float(2)) | (
        tx[:, categorical_column] == np.float(3)
    )

    return indices_cat_0, indices_cat_1, indices_cat_2_3


def divide_train_dataset_in_subsets(tx, y, ids, indices):
    """
    Function that actually divides train dataset according to the boolean masks passed as arguments
    Args:
        tx : array of shape (N,D)
        y :  array of shape (N,) containing the labels
        ids :  array of shape (N, ) containing ids of every observation
        indices :  boolean mask, array of shape (q, )
    
    Returns:
        tx[indices] :  array of shape (q,D)
        y[indices]  :  array of shape (q, )
        ids[indices] : array of shape (q,)
    
    """

    return tx[indices], y[indices], ids[indices]


def divide_test_dataset_in_subsets(tx, ids, indices):
    """
    Function that actually divides test dataset according to the boolean masks passed as arguments

    Args:
        tx : array of shape (N,D)
        ids :  array of shape (N, ) containing ids of every observation
        indices :  boolean mask, array of shape (q, )
    
    Returns:
        tx[indices] :  array of shape (q,D)
        ids[indices] : array of shape (q,)
    
    """

    return tx[indices], ids[indices]


def reordering_predictions(predictions, ids):
    """
    Helper function to reorder predictions given by each model before creating submission file
    
    Args:
    
        predictions :  array of shape (N,) containing predictions
        ids :  array of shape (N,) containing non sorted ids corresponding to each prediction
    
    Returns:
        predictions :  array of shape (N,) containing predictions sorted according to ids
        ids :          array of shape (N,) containing sorted array of ids
    """

    new_row_order = np.argsort(ids)
    return predictions[new_row_order], ids[new_row_order]
