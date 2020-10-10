#######################################################################
# IMPORTS
#######################################################################
import numpy as np
import dataLoader as dl
from LDA import myLDA
import MultiGaussian as mg
import ReadNormalizedDataset
#######################################################################
# CODE STARTS HERE
#######################################################################
def LDA2dGaussGM(num_crossval):
    if ((0 == num_crossval) or (1 == num_crossval)):
        print('Atleast 2-fold cross validation is required. Not supporting 0,1 fold...')
        return
    # Load Digits dataset
    digitsLoader = dl.digitsDataLoader()
    data = digitsLoader.getDataset()
    X = ReadNormalizedDataset.NormalizeDataset(data[:, 0:-1])
    X = X[:, ~np.all(X == 0, axis=0)]
    data = np.hstack((X, data[:, -1].reshape((X.shape[0], 1))))
    N = data.shape[0]
    blockSize = int((1/num_crossval)*N)
    err_arr_trn = np.zeros((num_crossval,))
    err_arr_val = np.zeros((num_crossval,))
    ###########################################################################
    # Training data and validation data separation
    ###########################################################################
    lda = myLDA()
    index = np.arange(N)
    for fold in range(num_crossval):
        if (fold == num_crossval - 1):
            v_idx = np.arange(fold * blockSize, N)
        else:
            v_idx = np.arange(fold*blockSize, (fold+1)*blockSize)
        t_idx = np.delete(index, v_idx)
        val_data = data[v_idx, :]
        trn_data = data[t_idx, :]
        N_val, D_val = val_data.shape
        D_val -= 1
        N_trn, D_trn = trn_data.shape
        D_trn -= 1
        ###########################################################################
        # Perform LDA
        ###########################################################################
        lda.loadData(trn_data)
        projectionMatrix = lda.performLDA()
        ###########################################################################
        # Project training and test data
        ###########################################################################
        w, e_val = lda.projectData(dimension=2)
        reduced_trn_data = trn_data[:, 0:D_trn] @ w
        reduced_val_data = val_data[:, 0:D_val] @ w
        D_trn_reduced = reduced_trn_data.shape[1]
        D_val_reduced = reduced_val_data.shape[1]
        reduced_trn_data = np.hstack((reduced_trn_data, trn_data[:, -1].reshape((trn_data.shape[0],1))))
        reduced_val_data = np.hstack((reduced_val_data, val_data[:, -1].reshape((val_data.shape[0],1))))
        ###########################################################################
        # Perform Gaussian generative modelling
        ###########################################################################
        multi_gauss = mg.multiGaussian()
        est_prior, est_mu, est_sigma = multi_gauss.performMLE(reduced_trn_data)
        prediction_trn = multi_gauss.classify(reduced_trn_data[:, 0:-1], est_prior, est_mu, est_sigma)
        prediction_val = multi_gauss.classify(reduced_val_data[:, 0:-1], est_prior, est_mu, est_sigma)
        err_val = (np.sum(prediction_val != val_data[:, -1])) * (100. / N_val)
        err_trn = (np.sum(prediction_trn != trn_data[:, -1])) * (100. / N_trn)
        err_arr_val[fold] = err_val
        err_arr_trn[fold] = err_trn
        print("LDA2dGaussGM: fold = ", fold, " training error rate = ", err_trn)
        print("LDA2dGaussGM: fold = ", fold, " validation error rate = ", err_val)
        print('')
        if (fold == num_crossval - 1):
            print('LDA2dGaussGM: average training error = ', np.mean(err_arr_trn))
            print('LDA2dGaussGM: average validation error = ', np.mean(err_arr_val))
            print('LDA2dGaussGM: standard deviation of training error = ', np.std(err_arr_trn))
            print('LDA2dGaussGM: standard deviation of validation error = ', np.std(err_arr_val))
        print('')

