#######################################################################
# PACKAGE IMPORTS
#######################################################################
import numpy as np
#######################################################################
# Function definitions
#######################################################################
def process_data(dataset, is_cat_conv_reqd = False, is_binary_classification=True):
    factor_max_limit = 6.
    # Number of features
    n_feature = dataset.shape[1]
    for feature in range(n_feature):
        if ((dataset[:, feature] == -1).any()):
            feature_mode = np.argmax(np.bincount(dataset[~(dataset[:, feature] == -1), feature]))
            dataset[(dataset[:, feature] == -1), feature] = feature_mode
        #######################################################################
        # Convert continuous variable to categorical
        #######################################################################
        if (True == is_cat_conv_reqd):
            if (np.unique(dataset[:, feature]).shape[0] >= factor_max_limit):
                split_point = np.mean(dataset[:, feature])
                dataset[:, feature] = convert_to_categorical(dataset[:, feature], split_point)
    if (True == is_binary_classification):
        labels = np.unique(dataset[:, -1])
        label_idx_neg = dataset[:, -1] == labels[0]
        label_idx_pos = dataset[:, -1] != labels[0]
        dataset[label_idx_neg, -1] = -1.
        dataset[label_idx_pos, -1] = 1.
    return dataset
#######################################################################
# This function converts a continuos valued variable to categorical
#######################################################################
def convert_to_categorical(dataset, split_point):
    category = np.zeros((dataset.shape[0],))
    category[dataset < split_point] = 0.
    category[dataset >= split_point] = 1.
    return category
#######################################################################
# This function returns bootstrapped samples
#######################################################################
def get_bootstrap_sample(dataset, size):
    n_sample = dataset.shape[0]
    bootstrap_idx = np.random.choice(np.arange(n_sample), size=size)
    return dataset[bootstrap_idx, :]
