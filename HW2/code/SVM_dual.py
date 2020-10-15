#######################################################################
# IMPORTS
#######################################################################
import numpy as np
import os
from dataloader import dataloader
import utils
import config
#######################################################################
# FUNCTION DEFINITIONS
#######################################################################
def SVM_dual(dataset):
    #######################################################################
    # Load dataset and get train and test data
    #######################################################################
    C_arr = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
    dl = dataloader(dataset)
    trn_data, test_data = dl.get_train_test_data(config.data_use_percent, config.train_split_percent)
    X_trn = trn_data[:, 0:-1]
    y_trn = trn_data[:, -1]
    y_trn[y_trn == 0] = -1
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1]
    y_test[y_test == 0] = -1
    N = X_trn.shape[0]
    N_test = X_test.shape[0]
    #######################################################################
    # Train SVM
    #######################################################################
    svm_objs = [utils.svm(kernel='linear', C=C_arr[c_idx]) for c_idx in range(len(C_arr))]
    index = np.arange(N)
    blockSize = int((1 / config.num_crossval) * N)
    err_arr_trn = np.zeros((len(C_arr), config.num_crossval))
    err_arr_val = np.zeros((len(C_arr), config.num_crossval))
    for c_idx in range(len(C_arr)):
        for fold in range(config.num_crossval):
            if (fold == config.num_crossval - 1):
                v_idx = np.arange(fold * blockSize, N)
            else:
                v_idx = np.arange(fold * blockSize, (fold + 1) * blockSize)
            t_idx = np.delete(index, v_idx)
            val_data = X_trn[v_idx, :]
            val_data_y = y_trn[v_idx]
            trn_data = X_trn[t_idx, :]
            trn_data_y = y_trn[t_idx]
            N_val = val_data.shape[0]
            N_trn = trn_data.shape[0]
            svm_objs[c_idx].fit(trn_data, trn_data_y)
            prediction_trn = svm_objs[c_idx].predict(trn_data)
            prediction_val = svm_objs[c_idx].predict(val_data)
            err_trn = (np.sum(prediction_trn != trn_data_y)) * (100. / N_trn)
            err_val = (np.sum(prediction_val != val_data_y)) * (100. / N_val)
            err_arr_trn[c_idx, fold] = err_trn
            err_arr_val[c_idx, fold] = err_val
            if (True == config.is_fold_verbose_enabled):
                print('SVM_dual: C = ', svm_objs[c_idx].C, ' fold = ', fold, ' training error rate = ', err_trn)
                print('SVM_dual: C = ', svm_objs[c_idx].C, ' fold = ', fold, ' validation error rate = ', err_val)
                print('')
        print('SVM_dual: Summary for C = ', svm_objs[c_idx].C)
        print('average training error = ', np.mean(err_arr_trn[c_idx, :]))
        print('average validation error = ', np.mean(err_arr_val[c_idx, :]))
        print('standard deviation of training error = ', np.std(err_arr_trn[c_idx, :]))
        print('standard deviation of validation error = ', np.std(err_arr_val[c_idx, :]))
        print('')
    C_optimal_idx = np.argmin(np.mean(err_arr_val, axis=1))
    C_optimal = C_arr[int(C_optimal_idx)]
    svm_optimal = svm_objs[int(C_optimal_idx)]
    prediction_test = svm_optimal.predict(X_test)
    err_test = (np.sum(prediction_test != y_test)) * (100. / N_test)
    print('##### SVM_dual: Summary #####')
    print('Optimal C = ', C_optimal)
    print('test error = ', err_test)