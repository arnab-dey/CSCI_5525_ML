#######################################################################
# IMPORTS
#######################################################################
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
import numpy as np
#######################################################################
# CLASS DEFINITIONS
#######################################################################
class bostonDataLoader:
    def __init__(self):
        self.X, self.t = load_boston(return_X_y=True)

    #######################################################################
    # This function returns required dataset passed as arguments
    #######################################################################
    def getDataset(self, dataset):
        if ((None is self.X) or (None is self.t)):
            print('invalid data')
            return None
        if (dataset == 'Boston50'):
            tau_50 = np.percentile(self.t, 50)
            y = np.zeros((self.t.shape[0]))
            y[:] = self.t[:]
            y[y < tau_50] = 0
            y[y >= tau_50] = 1
            return np.hstack((self.X, y.reshape((self.X.shape[0], 1))))
        elif (dataset == 'Boston75'):
            tau_75 = np.percentile(self.t, 75)
            y = np.zeros((self.t.shape[0]))
            y[:] = self.t[:]
            y[y < tau_75] = 0
            y[y >= tau_75] = 1
            return np.hstack((self.X, y.reshape((self.X.shape[0], 1))))
        else:
            print('invalid data request. Returning None...')
            return None

class digitsDataLoader:
    def __init__(self):
        self.X, self.t = load_digits(return_X_y=True)

    #######################################################################
    # This function returns the digits dataset
    #######################################################################
    def getDataset(self):
        return np.hstack((self.X, self.t.reshape((self.X.shape[0], 1))))


