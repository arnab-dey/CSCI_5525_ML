#######################################################################
# IMPORTS
#######################################################################
import numpy as np
import os
#######################################################################
# CLASS DEFINITIONS
#######################################################################
class dataloader:
    def __init__(self, dataset):
        ###########################################################################
        # Check if input files are present in the location
        ###########################################################################
        if not os.path.isfile(dataset):
            print("data data file can't be located")
            exit(1)
        self.data = np.genfromtxt(dataset, delimiter=',')
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


