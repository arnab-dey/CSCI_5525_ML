#######################################################################
# PACKAGE IMPORTS
#######################################################################
import numpy as np
import os
from tree import stump
import data_processor as dp
import matplotlib.pyplot as plt
import matplotlib as mpl
#######################################################################
# Static variable declaration
#######################################################################
isPlotReqd = True
isPlotPdf = True
train_percent = 0.8
################################################################################
# Settings for plot
################################################################################
if (True == isPlotReqd):
    if (True == isPlotPdf):
        mpl.use('pdf')
        fig_width  = 3.487
        fig_height = fig_width / 1.618
        rcParams = {
            'font.family': 'serif',
            'font.serif': 'Times',
            'text.usetex': True,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.labelsize': 8,
            'legend.fontsize': 8,
            'figure.figsize': [fig_width, fig_height]
           }
        plt.rcParams.update(rcParams)
#######################################################################
# Function definitions
#######################################################################
def rf(dataset):
    ###########################################################################
    # Check if dataset is present in the location
    ###########################################################################
    if not os.path.isfile(dataset):
        print("data data file can't be located")
        exit(1)
    data = np.genfromtxt(dataset, delimiter=',', dtype='int')
    data = dp.process_data(data)
    n_samples = data.shape[0]
    #######################################################################
    # Split into train and test
    #######################################################################
    np.random.shuffle(data)
    train_data = data[0:int(n_samples * train_percent), :]
    test_data = data[int(n_samples * train_percent):, :]
    n_trn = train_data.shape[0]
    n_tst = test_data.shape[0]
    B = 100 # Fixing this as required in HW. This is the number of trees in the forest
    eps = 1e-15 # Added to avoid division by 0 if required
    # Array of number of random feature values
    m_arr = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_error_arr = np.zeros((len(m_arr), B))
    test_error_arr = np.zeros((len(m_arr), B))
    #######################################################################
    # Run random forest for each values of m
    #######################################################################
    for m_idx in range(len(m_arr)):
        m = m_arr[m_idx]
        train_error, test_error = run_random_forest(train_data, test_data, B=B, m=m)
        train_error_arr[m_idx, :] = train_error[:]
        test_error_arr[m_idx, :] = test_error[:]

    #######################################################################
    # Plot of error rates: Q2.i: m=3
    #######################################################################
    if (True == isPlotReqd):
        ###########################################################################
        # Configure axis and grid
        ###########################################################################
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle="-.", linewidth='0.5')

        x_axis = np.linspace(1, B, num=int(B))
        m_idx = 1
        ax.plot(x_axis, train_error_arr[m_idx, :], label='Training error')
        ax.plot(x_axis, test_error_arr[m_idx, :], label='Test error')

        ax.set_xlabel(r'number of trees', fontsize=8)
        ax.set_ylabel(r'Error rate', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q2i_error_rate.pdf')
        else:
            plt.show()

    #######################################################################
    # Plot of error rates: Q2.ii
    #######################################################################
    if (True == isPlotReqd):
        ###########################################################################
        # Configure axis and grid
        ###########################################################################
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle="-.", linewidth='0.5')
        ax.plot(m_arr, train_error_arr[:, -1], label='Training error')
        ax.plot(m_arr, test_error_arr[:, -1], label='Test error')

        ax.set_xlabel(r'number of random features', fontsize=8)
        ax.set_ylabel(r'Error rate', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q2ii_error_rate.pdf')
        else:
            plt.show()

    #######################################################################
    # Plot of train error rates on same plot: Q2.ii
    #######################################################################
    if (True == isPlotReqd):
        ###########################################################################
        # Configure axis and grid
        ###########################################################################
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle="-.", linewidth='0.5')

        x_axis = np.linspace(1, B, num=int(B))
        for m_idx in range(len(m_arr)):
            plt_label = 'Training error: m=' + str(m_arr[m_idx])
            ax.plot(x_axis, train_error_arr[m_idx, :], label=plt_label)

        ax.set_xlabel(r'number of trees', fontsize=8)
        ax.set_ylabel(r'Error rate', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q2ii_error_rate_trn.pdf')
        else:
            plt.show()

    #######################################################################
    # Plot of train error rates on same plot: Q2.ii
    #######################################################################
    if (True == isPlotReqd):
        ###########################################################################
        # Configure axis and grid
        ###########################################################################
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle="-.", linewidth='0.5')

        x_axis = np.linspace(1, B, num=int(B))
        for m_idx in range(len(m_arr)):
            plt_label = 'Test error: m=' + str(m_arr[m_idx])
            ax.plot(x_axis, test_error_arr[m_idx, :], label=plt_label)

        ax.set_xlabel(r'number of trees', fontsize=8)
        ax.set_ylabel(r'Error rate', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q2ii_error_rate_tst.pdf')
        else:
            plt.show()

###########################################################################
# This function predicts samples in binary fashion in a random forest
###########################################################################
def predict_rf_binary(data, forest):
    prediction = np.zeros((data.shape[0], len(forest)))
    majority_label = np.zeros((data.shape[0],))
    for b in range(len(forest)):
        tree = forest[b]
        prediction[:, b] = tree.predict(data)
    ###########################################################################
    # Count majority vote for prediction
    ###########################################################################
    for sample in range(data.shape[0]):
        (values, counts) = np.unique(prediction[sample, :], return_counts=True)
        majority_label[sample] = values[np.argmax(counts)]

    return majority_label
###########################################################################
# This function runs random forest algorithm and returns train error rates
# and test error rate if test data is provided
###########################################################################
def run_random_forest(train_data, test_data=None, B=100, m=2):
    n_trn = train_data.shape[0]
    forest = []
    train_error_array = np.zeros((int(B),))
    test_error_array = np.zeros((int(B),))
    for tree_idx in range(int(B)):
        #######################################################################
        # Get bootstrapped sample
        #######################################################################
        train_data_boot = dp.get_bootstrap_sample(train_data, size=n_trn)
        #######################################################################
        # Randomly select m features
        #######################################################################
        feature_idx = np.random.choice(train_data_boot.shape[1] - 1, size=m, replace=False)
        X_train = train_data_boot[:, feature_idx]
        y_train = train_data_boot[:, -1]
        #######################################################################
        # Learn stump
        #######################################################################
        tree = stump(np.asarray(feature_idx))
        tree.train(X_train, y_train)
        #######################################################################
        # Append the tree to the forest
        #######################################################################
        forest.append(tree)
        #######################################################################
        # Predict and compute error rate for a single tree
        #######################################################################
        pred_train_tree = tree.predict(X_train, is_feat_map_reqd=False)
        error_train_tree = np.sum(y_train != pred_train_tree) / X_train.shape[0]
        if (None is not test_data):
            pred_test_tree = tree.predict(test_data[:, 0:-1], is_feat_map_reqd=True)
            error_test_tree = np.sum(test_data[:, -1] != pred_test_tree) / test_data.shape[0]

        #######################################################################
        # Predict and compute error rate for the forest
        #######################################################################
        pred_train_forest = predict_rf_binary(X_train, forest)
        error_train_forest = np.sum(y_train != pred_train_forest) / X_train.shape[0]
        train_error_array[tree_idx] = error_train_forest
        if (None is not test_data):
            pred_test_forest = predict_rf_binary(test_data[:, 0:-1], forest)
            error_test_forest = np.sum(test_data[:, -1] != pred_test_forest)/test_data.shape[0]
            test_error_array[tree_idx] = error_test_forest
        #######################################################################
        # Console log print
        #######################################################################
        print('RF: m = ', m, 'Total trees = ', tree_idx + 1, 'Training error rate = ', error_train_forest)
        if (None is not test_data):
            print('RF: m = ', m, 'Total trees = ', tree_idx + 1, 'Test error rate = ', error_train_forest)

    return train_error_array, test_error_array
