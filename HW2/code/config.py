#######################################################################
# CONFIG VARIABLES
#######################################################################
# This variable configures data percentage to be used by the algorithm
# SVM might take very long time with entire dataset.
# Use this variable to configure how much data to be used
# e.g. 1. means 100% i.e. entire dataset is used
# 0.1 means 10% of the dataset (randomly chosen) is used
# use lower value to accelerate the process
data_use_percent = 1.

# This variable configures how much data to be used for training
# set to 0.8 for 80-20 train-test split
train_split_percent = 0.8

# This variable configures number of folds in K-fold cross validation
# set to 10 for 10-fold CV
num_crossval = 10

# Set this variable to True to print error stats for each fold.
# For HW2, only average over K-folds is needed to be printed. Therefore,
# this variable is set to False by default.
# Use for debug purpose
is_fold_verbose_enabled = False