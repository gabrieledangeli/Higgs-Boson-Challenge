""" Python file to predict labels of unknown data """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from helpers import *
from implementations import *
from crossvalidation import *
from preprocessing import *
from dataset_splitting import *
from feature_engineering import *

# Loading train data

filename = "train.csv"
data_folder = "./data/"
file_path = data_folder + filename
y, tx, ids, features = load_train_data(file_path)

# Computing preprocessing routine

(
    list_subsets,
    list_features,
    y_0,
    y_1,
    y_2_3,
    columns_to_drop_in_subsets,
) = preprocessing(tx, y, ids, features)

# We want to introduce interaction factors between variables during the preprocessing routine.
# However, there is not statistical significance that multiplying variables with trigonometric functions
# may improve the model performance. Therefore, since trigonometric values are the last columns of each
# dataset, we save in a list how many columns are related to trigonometric values in each subset in order
# not to multiply columns with them later.

how_many_trig_features = [2, 1, 2]

# Standardizing data before using them

list_means = []
list_std = []
for idx in range(3):
    list_subsets[idx], mean, std = standardize(list_subsets[idx])
    list_means.append(mean)
    list_std.append(std)

# Defining optimal hyperparameters values (see notebook choosing_hyperparameters.ipynb for more details). The values obtained in the
# previous notebook have been rounded to the closest power of 10.

best_lambdas = [0.0001, 0.0001, 0.00001]
best_degrees = [7, 7, 7]

# Loading test data

filename = "test.csv"
data_folder = "./data/"
file_path = data_folder + filename
tx_test, test_ids, features_test = load_test_data(file_path)

# Retrieving logical masks to divide test dataset

categorical_column = np.where(features_test == "PRI_jet_num")[0][0]
mask_0_test, mask_1_test, mask_2_3_test = divide_indices_in_subsets(
    tx_test, categorical_column
)

# Removing categorical column since it is now useless

tx_test = np.delete(tx_test, categorical_column, axis=1)
# since we delete the column in tx, we also delete the name of the categorical feature used to divide the dataset
features_test = np.delete(features_test, categorical_column)

# Splitting test dataset, the output vector and ids w.r.t according to the mask

subset_0_test, ids_0_test = divide_test_dataset_in_subsets(
    tx_test, test_ids, mask_0_test
)
subset_1_test, ids_1_test = divide_test_dataset_in_subsets(
    tx_test, test_ids, mask_1_test
)
subset_2_3_test, ids_2_3_test = divide_test_dataset_in_subsets(
    tx_test, test_ids, mask_2_3_test
)

# Defining a list containing each subset

list_subsets_test = [subset_0_test, subset_1_test, subset_2_3_test]

# Define a list containing features for each subset

list_features_test = [features_test] * 3

# Dropping columns as done for train dataset and managing remaining missing values

for idx in range(3):
    list_subsets_test[idx] = list_subsets_test[idx][:, columns_to_drop_in_subsets[idx]]
    list_features_test[idx] = list_features_test[idx][columns_to_drop_in_subsets[idx]]
    for col in range(list_subsets_test[idx].shape[1]):
        median = np.nanmedian(list_subsets_test[idx][:, col])
        index = np.isnan(list_subsets_test[idx][:, col])
        list_subsets_test[idx][index, col] = median

# The last column in subset_0 is a zeros vector (see the documentation). Therefore, we drop it not to have problems when standardizing

list_subsets_test[0] = np.delete(list_subsets_test[0], -1, 1)
list_features_test[0] = np.delete(list_features_test[0], -1)

# Defining trigonometric features (sine and cosine) starting from columns related to angle values

columns_angles_0 = [11, 14, 16]
columns_angles_1 = [11, 14, 16, 20]
columns_angles_2 = [15, 18, 20, 27]

list_subsets_test[0], list_features_test[0] = trigonometrics(
    list_subsets_test[0], columns_angles_0, list_features_test[0]
)
list_subsets_test[1], list_features_test[1] = trigonometrics(
    list_subsets_test[1], columns_angles_1, list_features_test[1]
)
list_subsets_test[2], list_features_test[2] = trigonometrics(
    list_subsets_test[2], columns_angles_2, list_features_test[2]
)

# Applying logarithmic transformation to skewed distributions in each subset

to_log_c0 = [0, 1, 2, 3, 5, 6, 7, 9, 11, 13, 14]
to_log_c1 = [0, 1, 2, 3, 5, 6, 7, 9, 11, 13, 14, 15, 17]
to_log_c2 = [0, 1, 2, 3, 5, 8, 9, 10, 13, 15, 17, 18, 19, 22, 24]


list_subsets_test[0][:, to_log_c0] = log_transform(list_subsets_test[0][:, to_log_c0])
list_subsets_test[1][:, to_log_c1] = log_transform(list_subsets_test[1][:, to_log_c1])
list_subsets_test[2][:, to_log_c2] = log_transform(list_subsets_test[2][:, to_log_c2])

# Handling outliers by replacing them with 5% or 95% percentiles

for idx in range(3):
    list_subsets_test[idx] = capping_outliers(list_subsets_test[idx])

# Identifying useless variables and dropping corresponding columns in each subset

useless_c0 = [3, 5, 8, 16, 17, 19, 20]
useless_c1 = [11, 19, 20, 21, 22, 23, 24, 25]
useless_c2 = [15, 17, 18, 19, 21, 22, 26, 27, 28, 30, 31, 32]

# Dropping columns corresponding to useless variables in each subset

a = list(range(list_subsets_test[0].shape[1]))
useful_c0 = np.delete(a, useless_c0)
list_subsets_test[0] = list_subsets_test[0][:, useful_c0]
list_features_test[0] = list_features_test[0][useful_c0]

b = list(range(list_subsets_test[1].shape[1]))
useful_c1 = np.delete(b, useless_c1)
list_subsets_test[1] = list_subsets_test[1][:, useful_c1]
list_features_test[1] = list_features_test[1][useful_c1]

c = list(range(list_subsets_test[2].shape[1]))
useful_c2 = np.delete(c, useless_c2)
list_subsets_test[2] = list_subsets_test[2][:, useful_c2]
list_features_test[2] = list_features_test[2][useful_c2]

# Standardizing test data

for idx in range(3):
    list_subsets_test[idx] = (list_subsets_test[idx] - list_means[idx]) / list_std[idx]

# Expanding both test and train dataset according to degrees in best_degrees

for idx in range(3):
    list_subsets[idx] = build_poly(
        list_subsets[idx], best_degrees[idx], how_many_trig_features[idx]
    )
    list_subsets_test[idx] = build_poly(
        list_subsets_test[idx], best_degrees[idx], how_many_trig_features[idx]
    )

# Training ridge regression model using train data for each subset

final_ws = [0] * 3
final_ws[0], _ = ridge_regression(y_0, list_subsets[0], best_lambdas[0])
final_ws[1], _ = ridge_regression(y_1, list_subsets[1], best_lambdas[1])
final_ws[2], _ = ridge_regression(y_2_3, list_subsets[2], best_lambdas[2])

# Computing predictions for each test subset

prediction_0 = predict_ridge(list_subsets_test[0], final_ws[0])
prediction_1 = predict_ridge(list_subsets_test[1], final_ws[1])
prediction_2 = predict_ridge(list_subsets_test[2], final_ws[2])
all_predictions = np.concatenate([prediction_0, prediction_1, prediction_2])
all_ids = np.concatenate([ids_0_test, ids_1_test, ids_2_3_test])
all_predictions, all_ids = reordering_predictions(all_predictions, all_ids)

# Creating final submission

create_submission(
    all_ids, all_predictions, ["Id", "Prediction"], "./ridge_regression_final.csv"
)
