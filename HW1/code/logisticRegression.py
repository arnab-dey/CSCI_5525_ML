#######################################################################
# IMPORTS
#######################################################################
import numpy as np
import dataLoader as dl
import ReadNormalizedDataset
#######################################################################
# Variable declaration
#######################################################################
isPrintReqd = True # This is to enable/disable console print
#######################################################################
# CODE STARTS HERE
#######################################################################
class discriminator:
    def __init__(self):
        self.data = None

    #######################################################################
    # This function loads dataset and normalizes it
    #######################################################################
    def loadData(self, data, isNormalizationReqd = True):
        if (data == 'Boston50'):
            bostonLoader = dl.bostonDataLoader()
            data = bostonLoader.getDataset('Boston50')
        elif (data == 'Boston75'):
            bostonLoader = dl.bostonDataLoader()
            data = bostonLoader.getDataset('Boston75')
        else:
            digitsLoader = dl.digitsDataLoader()
            data = digitsLoader.getDataset()
        if (isNormalizationReqd == True):
            # Normalize dataset
            X = ReadNormalizedDataset.NormalizeDataset(data[:,0:-1])
            X = X[:, ~np.all(X == 0, axis=0)]
            self.data = np.hstack((X, data[:,-1].reshape((X.shape[0], 1))))
        else:
            self.data = data

    #######################################################################
    # This function returns class specific random 80-20 split of
    # whole dataset to training and test data
    #######################################################################
    def getSplitData(self):
        num_class = int(np.max(self.data[:, -1])) + 1
        trn_data = []
        val_data = []
        for classCount in range(num_class):
            class_data = self.data[self.data[:, -1] == np.asarray(classCount)]
            class_size = np.size(class_data, 0)
            np.random.shuffle(class_data)
            trn_data.append(class_data[0:int(0.8*class_size),:])
            val_data.append(class_data[int(0.8*class_size):,:])
        trn_data = np.concatenate(trn_data)
        val_data = np.concatenate(val_data)
        np.random.shuffle(trn_data)
        np.random.shuffle(val_data)
        return trn_data, val_data

    #######################################################################
    # This function reformats class labels
    # Columns corresponding to a label gets 1 while all other columns
    # become 0. For e.g. in case of digits dataset, data labeled 3 will
    # be reformatted to an array whose 3rd element is 1 and all other
    # elements are 0.
    #######################################################################
    def getReformattedLabels(self, y_n):
        num_class = int(np.max(y_n)) + 1
        r_nk = np.zeros((y_n.shape[0], num_class))
        for sample in range(y_n.shape[0]):
            r_nk[sample, int(y_n[sample])] = 1
        return r_nk

    #######################################################################
    # This function calculates softmax output
    #######################################################################
    def getOutputSoftmax(self, o):
        exp_o = np.exp(o)
        sum_exp_o = np.sum(exp_o, axis=1)
        y = exp_o.T / sum_exp_o
        return y.T

    #######################################################################
    # This function performs gradient descent
    # Learning rate, eta, and regularization parameter, C, are
    # passed as arguments
    # r_nk is reformatted true labels returned from 'getReformattedLabels'
    # x_nd are features
    # This function returns learned weights values
    #######################################################################
    def performGradientDescent(self, x_nd, r_nk, eta=0.001, C=10):
        num_class = int(r_nk.shape[1])
        N, D = x_nd.shape
        ###########################################################################
        # Initialize weights
        ###########################################################################
        # Random number within [-0.01,0.01]
        w_dk = ((np.random.rand(D, num_class) * 2) - 1) * 0.01
        ###########################################################################
        # Define parameters
        ###########################################################################
        isConverged = False
        prev_loss = 0.
        loss = 0.
        loss_change_thr = 0.001
        patience_count = 0
        # If the difference between current and previous error is less than
        # 'loss_change_thr' for 5 consecutive iterations, we assume convergence
        patience_limit = 5
        while(isConverged == False):
            prev_loss = loss
            loss = 0.
            o_nk = x_nd @ w_dk
            y_nk = self.getOutputSoftmax(o_nk)
            ###########################################################################
            # Calculate updates
            ###########################################################################
            del_w_dk = eta * (x_nd.T @ (r_nk - y_nk)) - eta * C * w_dk
            w_dk += del_w_dk
            ###########################################################################
            # Calculate loss
            ###########################################################################
            loss = -np.sum(np.diag(r_nk.T @ np.log(y_nk)))
            ###########################################################################
            # Check for convergence
            ###########################################################################
            if (np.abs(loss - prev_loss) <= loss_change_thr):
                patience_count += 1
                if (patience_count == patience_limit):
                    isConverged = True
        return w_dk

    ###########################################################################
    # This function performs logistic regression required by question 4
    ###########################################################################
    def LogisticRegression(self, num_splits, train_percent):
        ###########################################################################
        # Extract information from data
        ###########################################################################
        N, D = self.data.shape
        # Last column represents class label, therefore feature dimension is D-1
        D -= 1
        num_class = int(np.max(self.data[:, -1])) + 1
        err_trn_arr = np.zeros((num_splits, np.asarray(train_percent).shape[0]))
        err_val_arr = np.zeros((num_splits, np.asarray(train_percent).shape[0]))
        for itr in range(num_splits):
            # Get random 80-20 split for train-validation data
            trn_data, val_data = self.getSplitData()
            N_trn = trn_data.shape[0]
            N_val = val_data.shape[0]
            # extract percentage of training data to be used
            percent_idx = 0
            for percent in train_percent:
                trn_data_subset = trn_data[0:int(percent*N_trn/100),:]
                N_trn_subset = trn_data_subset.shape[0]
                x_nd = trn_data_subset[:,0:-1]
                y_n = trn_data_subset[:,-1]
                # Append ones to represent x_0
                x_nd = np.hstack((np.ones((x_nd.shape[0], 1)), x_nd))
                # Reformat labels for all traning data
                r_nk = self.getReformattedLabels(y_n)
                w_dk = self.performGradientDescent(x_nd, r_nk)
                ###########################################################################
                # Calculate error rate on training data
                ###########################################################################
                o_nk = x_nd @ w_dk
                y_nk = self.getOutputSoftmax(o_nk)
                y_predicted = np.argmax(y_nk, axis=1)
                err_trn = (np.sum(y_predicted != y_n)) * (100. / N_trn_subset)
                ###########################################################################
                # Calculate error rate on validation data
                ###########################################################################
                xVal_nd = np.hstack((np.ones((N_val, 1)), val_data[:, 0:-1]))
                o_nk = xVal_nd @ w_dk
                y_nk = self.getOutputSoftmax(o_nk)
                y_predicted = np.argmax(y_nk, axis=1)
                err_val = (np.sum(y_predicted != val_data[:, -1])) * (100. / N_val)
                ###########################################################################
                # Store error rates
                ###########################################################################
                err_trn_arr[itr, percent_idx] = err_trn
                err_val_arr[itr, percent_idx] = err_val
                # TODO: Remove later
                # LR_model = LogisticRegression(C=10, solver='lbfgs', max_iter=1000, multi_class='multinomial')
                # LR_model.fit(trn_data_subset[:,0:-1], trn_data_subset[:,-1])
                # sk_y_predicted = LR_model.predict(val_data[:,0:-1])
                # sk_err_val = (np.sum(sk_y_predicted != val_data[:,-1])) * (100. / N_val)
                # print('percentage data = ', percent, ' Error on sk learn validation data = ', sk_err_val)
                percent_idx += 1
                ###########################################################################
                # Console print for each train percent
                ###########################################################################
                if (True == isPrintReqd):
                    print('LR: Iteration ', itr)
                    print('LR: Training data ', percent, '%')
                    print('LR: Error on training data = ', err_trn)
                    print('LR: Error on validation data = ', err_val)
                    print('')
            ###########################################################################
            # Console print for each num_splits
            ###########################################################################
            if (True == isPrintReqd):
                print('LR: Iteration ', itr)
                print('### ERROR OVER ALL TRAINING DATA PERCENTAGES ###')
                print('LR: Mean of training error = ', np.mean(err_trn_arr[itr,:]))
                print('LR: Std deviation of training error = ', np.std(err_trn_arr[itr,:]))
                print('LR: Mean of validation error = ', np.mean(err_val_arr[itr,:]))
                print('LR: Std deviation of validation error = ', np.std(err_val_arr[itr,:]))
                print('')
        ###########################################################################
        # Console print for all iterations
        ###########################################################################
        if (True == isPrintReqd):
            for percent in range(len(train_percent)):
                print('### ERROR OVER ALL SPLIT ITERATIONS ###')
                print('LR: Training data ', train_percent[percent], '%')
                print('LR: Mean of training error = ', np.mean(err_trn_arr[:, percent]))
                print('LR: Std deviation of training error = ', np.std(err_trn_arr[:, percent]))
                print('LR: Mean of validation error = ', np.mean(err_val_arr[:, percent]))
                print('LR: Std deviation of validation error = ', np.std(err_val_arr[:, percent]))
                print('')
        return err_trn_arr, err_val_arr