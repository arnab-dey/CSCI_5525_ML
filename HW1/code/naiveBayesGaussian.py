#######################################################################
# IMPORTS
#######################################################################
import numpy as np
import dataLoader as dl
import ReadNormalizedDataset
import MultiGaussian as mg
#######################################################################
# Variable declaration
#######################################################################
isPrintReqd = True # This is to enable/disable console print
#######################################################################
# Class definition
#######################################################################
class naiveBayes:
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

    ###########################################################################
    # This function performs Naive Bayes classification required by question 4
    ###########################################################################
    def naiveBayesGaussian(self, num_splits, train_percent):
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
                trn_data_subset = trn_data[0:int(percent * N_trn / 100), :]
                N_trn_subset = trn_data_subset.shape[0]
                ###########################################################################
                # Perform MLE first to estimate mu, sigma and priors
                ###########################################################################
                multi_gauss = mg.multiGaussian()
                est_prior, est_mu, est_sigma = multi_gauss.performMLE(trn_data_subset)
                ###########################################################################
                # We will use shared cov matrix by data pooling
                ###########################################################################
                est_shared_sigma = np.zeros((est_sigma[0, :, :].shape))
                for classCount in range(num_class):
                    est_shared_sigma += est_prior[classCount]*est_sigma[classCount, :, :]
                ###########################################################################
                # We will use Naive Bayes i.e. we wil use diagonal shared cov matrix
                ###########################################################################
                est_shared_sigma = np.diag(np.diag(est_shared_sigma))
                prediction_trn = multi_gauss.classify(trn_data_subset[:, 0:-1], est_prior, est_mu, est_shared_sigma,
                                                      isSharedSigma=True, isNaiveBayes=True)
                prediction_val = multi_gauss.classify(val_data[:, 0:-1], est_prior, est_mu, est_shared_sigma,
                                                      isSharedSigma=True, isNaiveBayes=True)
                err_val = (np.sum(prediction_val != val_data[:, -1])) * (100. / N_val)
                err_trn = (np.sum(prediction_trn != trn_data_subset[:, -1])) * (100. / N_trn_subset)
                err_trn_arr[itr, percent_idx] = err_trn
                err_val_arr[itr, percent_idx] = err_val
                percent_idx += 1
                ###########################################################################
                # Console print for each train percent
                ###########################################################################
                if (True == isPrintReqd):
                    print('GNB: Iteration ', itr)
                    print('GNB: Training data ', percent, '%')
                    print('GNB: Error on training data = ', err_trn)
                    print('GNB: Error on validation data = ', err_val)
                    print('')
            ###########################################################################
            # Console print for each num_splits
            ###########################################################################
            if (True == isPrintReqd):
                print('GNB: Iteration ', itr)
                print('### ERROR OVER ALL TRAINING DATA PERCENTAGES ###')
                print('GNB: Mean of training error = ', np.mean(err_trn_arr[itr, :]))
                print('GNB: Std deviation of training error = ', np.std(err_trn_arr[itr, :]))
                print('GNB: Mean of validation error = ', np.mean(err_val_arr[itr, :]))
                print('GNB: Std deviation of validation error = ', np.std(err_val_arr[itr, :]))
                print('')
        ###########################################################################
        # Console print for all iterations
        ###########################################################################
        if (True == isPrintReqd):
            for percent in range(len(train_percent)):
                print('### ERROR OVER ALL SPLIT ITERATIONS ###')
                print('GNB: Training data ', train_percent[percent], '%')
                print('GNB: Mean of training error = ', np.mean(err_trn_arr[:, percent]))
                print('GNB: Std deviation of training error = ', np.std(err_trn_arr[:, percent]))
                print('GNB: Mean of validation error = ', np.mean(err_val_arr[:, percent]))
                print('GNB: Std deviation of validation error = ', np.std(err_val_arr[:, percent]))
                print('')
        return err_trn_arr, err_val_arr
