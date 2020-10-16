Name: Arnab Dey
Student ID: 5563169
email: dey00011@umn.edu

#################################################################
# INSTRUCTION TO RUN CODE
#################################################################
Q2: Run 'script_2.py' to run the code for Q2
Q3: Run 'script_3.py' to run the code for Q3
Q4: Run 'script_4.py' to run the code for Q4

#################################################################
# ADDITIONAL DETAILS
#################################################################
1. I have taken whole data set for all the problems i.e. 100% of the provided data is used.
2. Implementation of SVM algorithm is in 'utils.py'
3. Code for loading the datasets is in 'dataloader.py'
4. I have used 'mfeat-fou' dataset for Q4.
5. Class 'dataloader' in 'dataloader.py' can be initialized with 'is_mfeat' argument. 'is_mfeat' is set to True if any of 'mfeat' dataset
	is being loaded. This is required as 'mfeat' data does not have labels added to the dataset. I add the labels in my code.
6. Console output for Q2 is in 'q2_console_log.txt'
7. Console output for Q3 is in 'q3_console_log.txt'
8. Console output for Q4 is in 'q4_console_log.txt'
9. 'config.py' has the config variables which can be used to choose how much data is to be used (to save cvxopt time), train-test
	split percentage, number of folds in cross-validation and verbose.

#################################################################
# HOW TO USE config.py
#################################################################
data_use_percent = 1.

This variable configures data percentage to be used by the algorithm.
SVM might take very long time with entire dataset.
Use this variable to configure how much data to be used
e.g. 1. means 100% i.e. entire dataset is used
0.1 means 10% of the dataset (randomly chosen) is used.
use lower value to accelerate the process.

train_split_percent = 0.8

This variable configures how much data to be used for training.
set to 0.8 for 80-20 train-test split.

num_crossval = 10

This variable configures number of folds in K-fold cross validation.
set to 10 for 10-fold CV.

is_fold_verbose_enabled = False

Set this variable to True to print error stats for each fold.
For HW2, only average over K-folds is needed to be printed. Therefore,
this variable is set to False by default.
Use for debug purpose.

THE HOMEWORK HAS BEEN DONE COMPLETELY ON MY OWN.