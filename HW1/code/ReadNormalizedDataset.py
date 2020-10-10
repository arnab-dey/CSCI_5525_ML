#######################################################################
# IMPORTS
#######################################################################
import numpy as np
#######################################################################
# This function standardizes given dataset
#######################################################################
def NormalizeDataset(dataset):
    # prepare training data output
    X_trn = dataset

    # mean and std
    mu = np.mean(X_trn, axis=0)
    mu[mu<1e-8] = 0
    s = np.std(X_trn, axis=0)
    s[s < 1e-8] = 0
    
    # normalize data
    X_test = (X_trn - np.tile(mu, (np.shape(X_trn)[0], 1)))
    X_trn_norm = (X_trn - np.tile(mu, (np.shape(X_trn)[0], 1))) / np.tile(s, (np.shape(X_trn)[0], 1))
    
    # fix nan
    X_trn_norm[np.isnan(X_trn_norm)]=0
    X_trn_norm[np.isinf(X_trn_norm)]=0

    # Remove non changing features
    # np.all(X_trn_norm)
    
    return X_trn_norm