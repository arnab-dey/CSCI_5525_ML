#######################################################################
# IMPORTS
#######################################################################
import numpy as np
import os
from dataloader import dataloader
import utils
#######################################################################
# VARIABLE DEFINITIONS
#######################################################################
is_fold_verbose_enabled = False
num_crossval = 10
#######################################################################
# FUNCTION DEFINITIONS
#######################################################################
def run_svm(svm_obj, X, y):
    N = X.shape[0]
    index = np.arange(N)
    blockSize = int((1 / num_crossval) * N)
    err_arr_trn = np.zeros((num_crossval,))
    err_arr_val = np.zeros((num_crossval,))
    for fold in range(num_crossval):
        if (fold == num_crossval - 1):
            v_idx = np.arange(fold * blockSize, N)
        else:
            v_idx = np.arange(fold * blockSize, (fold + 1) * blockSize)
        t_idx = np.delete(index, v_idx)
        val_data = X[v_idx, :]
        val_data_y = y[v_idx]
        trn_data = X[t_idx, :]
        trn_data_y = y[t_idx]
        N_val = val_data.shape[0]
        N_trn = trn_data.shape[0]
        svm_obj.fit(trn_data, trn_data_y)
        prediction_trn = svm_obj.predict(trn_data)
        prediction_val = svm_obj.predict(val_data)
        err_trn = (np.sum(prediction_trn != trn_data_y)) * (100. / N_trn)
        err_val = (np.sum(prediction_val != val_data_y)) * (100. / N_val)
        err_arr_trn[fold] = err_trn
        err_arr_val[fold] = err_val
        if (True == is_fold_verbose_enabled):
            print('kernel_SVM: C = ', svm_obj.C, ' fold = ', fold, ' training error rate = ', err_trn)
            print('kernel_SVM: C = ', svm_obj.C, ' fold = ', fold, ' validation error rate = ', err_val)
            print('')
    return err_arr_trn, err_arr_val

def kernel_SVM(dataset):
    #######################################################################
    # Load dataset and get train and test data
    #######################################################################
    C_arr = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
    use_percent = 0.1
    train_split_percent = 0.8
    dl = dataloader(dataset)
    trn_data, test_data = dl.get_train_test_data(use_percent, train_split_percent)
    X_trn = trn_data[:, 0:-1]
    y_trn = trn_data[:, -1]
    y_trn[y_trn == 0] = -1
    X_test = test_data[:, 0:-1]
    y_test = test_data[:, -1]
    y_test[y_test == 0] = -1
    N = X_trn.shape[0]
    N_test = X_test.shape[0]
    gamma_init = (1./(X_trn.shape[0] * X_trn.var()))*1e-2
    # gamma_init = 1e-2
    num_gamma = 5
    hyper_params = []
    for c_idx in range(len(C_arr)):
        for g_idx in range(num_gamma):
            hyper_params.append(np.array([C_arr[c_idx], gamma_init*10**(g_idx)]))
    #######################################################################
    # Train SVM
    #######################################################################
    svm_objs_rbf = [utils.svm(kernel='rbf', C=hyper_params[idx][0], gamma=hyper_params[idx][1]) for idx in range(len(hyper_params))]
    svm_objs_linear = [utils.svm(kernel='linear', C=C_arr[idx]) for idx in range(len(C_arr))]
    err_arr_trn_rbf = np.zeros((len(hyper_params), num_crossval))
    err_arr_trn_lin = np.zeros((len(C_arr), num_crossval))
    err_arr_val_rbf = np.zeros((len(hyper_params), num_crossval))
    err_arr_val_lin = np.zeros((len(C_arr), num_crossval))
    #######################################################################
    # Train SVM: RBF
    #######################################################################
    for idx in range(len(svm_objs_rbf)):
        err_trn, err_val = run_svm(svm_objs_rbf[idx], X_trn, y_trn)
        err_arr_trn_rbf[idx, :] = err_trn
        err_arr_val_rbf[idx, :] = err_val
        print('RBF kernel_SVM: Summary for C = ', svm_objs_rbf[idx].C, ' gamma = ', svm_objs_rbf[idx].gamma)
        print('RBF kernel_SVM: average training error = ', np.mean(err_trn))
        print('RBF kernel_SVM: average validation error = ', np.mean(err_val))
        print('RBF kernel_SVM: standard deviation of training error = ', np.std(err_trn))
        print('RBF kernel_SVM: standard deviation of validation error = ', np.std(err_val))
        print('')
    optimal_idx = np.argmin(np.mean(err_arr_val_rbf, axis=1))
    # C_optimal = C_arr[int(C_optimal_idx)]
    svm_optimal = svm_objs_rbf[int(optimal_idx)]
    prediction_test = svm_optimal.predict(X_test)
    err_test = (np.sum(prediction_test != y_test)) * (100. / N_test)
    print('##### RBF kernel_SVM: Summary #####')
    print('RBF kernel_SVM: Optimal C = ', svm_optimal.C, ', optimal gamma = ', svm_optimal.gamma)
    print('RBF kernel_SVM: test error = ', err_test)
    print('')
    #######################################################################
    # Train SVM: Linear
    #######################################################################
    for idx in range(len(C_arr)):
        err_trn, err_val = run_svm(svm_objs_linear[idx], X_trn, y_trn)
        err_arr_trn_lin[idx, :] = err_trn
        err_arr_val_lin[idx, :] = err_val
        print('Linear kernel_SVM: Summary for C = ', svm_objs_linear[idx].C)
        print('Linear kernel_SVM: average training error = ', np.mean(err_trn))
        print('Linear kernel_SVM: average validation error = ', np.mean(err_val))
        print('Linear kernel_SVM: standard deviation of training error = ', np.std(err_trn))
        print('Linear kernel_SVM: standard deviation of validation error = ', np.std(err_val))
        print('')
    optimal_idx = np.argmin(np.mean(err_arr_val_lin, axis=1))
    # C_optimal = C_arr[int(C_optimal_idx)]
    svm_optimal = svm_objs_linear[int(optimal_idx)]
    prediction_test = svm_optimal.predict(X_test)
    err_test = (np.sum(prediction_test != y_test)) * (100. / N_test)
    print('##### Linear kernel_SVM: Summary #####')
    print('Linear kernel_SVM: Optimal C = ', svm_optimal.C)
    print('Linear kernel_SVM: test error = ', err_test)