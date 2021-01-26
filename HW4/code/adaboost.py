#######################################################################
# PACKAGE IMPORTS
#######################################################################
import numpy as np
import os
from tree import stump
import data_processor as dp
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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
def adaboost(dataset):
    ###########################################################################
    # Check if dataset is present in the location
    ###########################################################################
    if not os.path.isfile(dataset):
        print("data data file can't be located")
        exit(1)
    data = np.genfromtxt(dataset, delimiter=',', dtype='int')
    #######################################################################
    # Preprocess data to handle missing values
    #######################################################################
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
    n_weak_learners = 100. # Fixing this as required in HW
    eps = 1e-8
    #######################################################################
    # Iniitialize weights
    #######################################################################
    D = np.ones((n_trn,))*(1/n_trn)
    alpha = np.zeros((int(n_weak_learners),))
    learners = []
    train_error_array = np.zeros((int(n_weak_learners),))
    test_error_array = np.zeros((int(n_weak_learners),))
    #######################################################################
    # Run Adaboost
    #######################################################################
    for learner_idx in range(int(n_weak_learners)):
        #######################################################################
        # Sample based on weights
        #######################################################################
        weighted_idx = np.random.choice(np.arange(n_trn), size=n_trn, p=D)
        X_train = train_data[weighted_idx, 0:-1]
        y_train = train_data[weighted_idx, -1]
        #######################################################################
        # Create and train stump
        #######################################################################
        weak_learner = stump()
        weak_learner.train(X_train, y_train)
        # weak_learner = DecisionTreeClassifier(criterion="entropy", max_depth=1)
        # weak_learner.fit(train_data[:, 0:-1], train_data[:, -1], sample_weight=D)
        # weak_learner.fit(X_train, y_train)
        #######################################################################
        # Add stump to array
        #######################################################################
        learners.append(weak_learner)
        #######################################################################
        # Calculate error for a single stump
        #######################################################################
        pred = weak_learner.predict(X_train)
        error = np.sum(D[y_train != pred])
        #######################################################################
        # Avoid division by 0 or log(0)
        #######################################################################
        # error_ratio = (1-error)/error
        if (error <= eps):
            error = eps
            error_ratio = (1 - error) / error
        elif (error == 1.):
            error_ratio = eps
        else:
            error_ratio = (1-error)/error
        #######################################################################
        # Compute alpha
        #######################################################################
        alpha_t = 0.5 * np.log(error_ratio)
        alpha[learner_idx] = alpha_t
        #######################################################################
        # Update sample weights
        #######################################################################
        D = D * np.exp(-alpha_t * y_train * pred)
        D = D / np.sum(D)
        #######################################################################
        # Calculate Adaboost error rate
        #######################################################################
        pred_train = predict_adaboost_binary(X_train, learners, alpha)
        pred_test = predict_adaboost_binary(test_data[:, 0:-1], learners, alpha)
        error_train = np.sum(y_train != pred_train)/n_trn
        error_test = np.sum(test_data[:, -1] != pred_test)/n_tst
        train_error_array[learner_idx] = error_train
        test_error_array[learner_idx] = error_test
        #######################################################################
        # Console log print
        #######################################################################
        print('Total learners = ', learner_idx+1, 'Training error rate = ', error_train)
        print('Total learners = ', learner_idx+1, 'Test error rate = ', error_test)

    #######################################################################
    # Plot of error rate
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

        x_axis = np.linspace(1, n_weak_learners, num=int(n_weak_learners))
        ax.plot(x_axis, train_error_array, label='Training error')
        ax.plot(x_axis, test_error_array, label='Test error')

        ax.set_xlabel(r'number of weak learners', fontsize=8)
        ax.set_ylabel(r'Error rate', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q1_error_rate.pdf')
        else:
            plt.show()
#######################################################################
# This function predicts from Adaboosted learners
#######################################################################
def predict_adaboost_binary(data, learners, alpha):
    prediction = np.zeros((data.shape[0],))
    for t in range(len(learners)):
        learner = learners[t]
        y_t = learner.predict(data)
        prediction += alpha[t] * y_t
    return np.sign(prediction)