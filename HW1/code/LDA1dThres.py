#######################################################################
# IMPORTS
#######################################################################
import numpy as np
import dataLoader as dl
from LDA import myLDA
import ReadNormalizedDataset
#######################################################################
# CODE STARTS HERE
#######################################################################
def LDA1dThres(num_crossval):
    if ((0 == num_crossval) or (1 == num_crossval)):
        print('Atleast 2-fold cross validation is required. Not supporting 0,1 fold...')
        return
    # Load Boston50 dataset
    bostonLoader = dl.bostonDataLoader()
    data = bostonLoader.getDataset('Boston50')
    # np.random.shuffle(data)
    # X = ReadNormalizedDataset.NormalizeDataset(data[:, 0:-1])
    # X = X[:, ~np.all(X == 0, axis=0)]
    # data = np.hstack((X, data[:, -1].reshape((X.shape[0], 1))))
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
        if (fold == num_crossval-1):
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
        w, e_val = lda.projectData(dimension=1)
        reduced_trn_data = trn_data[:, 0:D_trn] @ w
        reduced_val_data = val_data[:, 0:D_val] @ w
        D_trn_reduced = reduced_trn_data.shape[1]
        D_val_reduced = reduced_val_data.shape[1]
        ###########################################################################
        # Threshold choice
        # For two class case, the threshold that minimizes the error is the
        # projected value of the mean of total training data
        ###########################################################################
        threshold = w.T @ np.mean(trn_data[:, 0:D_trn], axis=0)
        # threshold = np.mean(reduced_trn_data)
        prediction_val = lda.predict(reduced_val_data, threshold)
        prediction_trn = lda.predict(reduced_trn_data, threshold)
        ###########################################################################
        # cross-validation data performance check
        ###########################################################################
        err_val = (np.sum(prediction_val != val_data[:, -1])) * (100. / N_val)
        err_trn = (np.sum(prediction_trn != trn_data[:, -1])) * (100. / N_trn)
        err_arr_val[fold] = err_val
        err_arr_trn[fold] = err_trn
        print("LDA1dThres: fold = ", fold, " training error rate = ", err_trn)
        print("LDA1dThres: fold = ", fold, " validation error rate = ", err_val)
        print('')
        if (fold == num_crossval-1):
            print('LDA1dThres:: Summary')
            print('average training error = ', np.mean(err_arr_trn))
            print('average validation error = ', np.mean(err_arr_val))
            print('standard deviation of training error = ', np.std(err_arr_trn))
            print('standard deviation of validation error = ', np.std(err_arr_val))
        print('')
