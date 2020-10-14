#######################################################################
# IMPORTS
#######################################################################
import numpy as np
import os
#######################################################################
# CLASS DEFINITIONS
#######################################################################
class dataloader:
    def __init__(self, dataset, is_mfeat=False):
        ###########################################################################
        # Check if input files are present in the location
        ###########################################################################
        if not os.path.isfile(dataset):
            print("data data file can't be located")
            exit(1)
        if (False == is_mfeat):
            self.data = np.genfromtxt(dataset, delimiter=',')
        else:
            self.data = np.loadtxt(dataset)
        self.is_mfeat = is_mfeat
        self.trn_data = None
        self.test_data = None

    #######################################################################
    # This function returns required dataset passed as arguments
    #######################################################################
    def get_train_test_data(self, use_percent=1., train_split_percent=0.8):
        if (None is self.data):
            print('invalid data')
            return None
        N = self.data.shape[0]
        if (True == self.is_mfeat):
            # Need to append class labels at last
            samples_per_digit = 200
            self.data = np.hstack((self.data, np.zeros((N, 1))))
            for digit in range(10):
                start = samples_per_digit*digit
                end = start+samples_per_digit
                self.data[start:end, -1] = digit
        # Take percentage of data specified in use_percent
        N_used = int(N*use_percent)
        index = np.arange(N)
        np.random.seed(2)
        np.random.shuffle(index)
        used_data = self.data[index[0:N_used],:]
        # Now split the data into train and test set
        self.trn_data = used_data[0:int(N_used*train_split_percent), :]
        self.test_data = used_data[int(N_used*train_split_percent):, :]
        return self.trn_data, self.test_data


