#######################################################################
# IMPORTS
#######################################################################
import numpy as np
#######################################################################
# CLASS DEFINITIONS
#######################################################################
class myLDA:
    def __init__(self):
        self.data = None
        self.w = None
        self.projection = None
        self.mean = None
        self.Sw = None
        self.Sb = None

    def loadData(self, data):
        self.data = data
    #######################################################################
    # This function computes the within class and between class
    # covariance matrices and returns the full projection matrix
    #######################################################################
    def performLDA(self):
        ###########################################################################
        # Extract information from data
        ###########################################################################
        N, D = self.data.shape
        # Last column represents class label, therefore feature dimension is D-1
        D -= 1
        ###########################################################################
        # Separate each class data and extract useful information
        ###########################################################################
        sep_data = []
        sep_num_data = []
        sep_mean_data = []
        sep_within_cls_sctr = []
        num_class = int(np.max(self.data[:, D])) + 1
        for classCount in range(num_class):
            class_data = self.data[self.data[:, D] == np.asarray(classCount)]
            class_size = np.size(class_data, 0)
            class_mean = np.mean(class_data[:, 0:D], axis=0)
            class_diff_from_mean = class_data[:, 0:D] - class_mean
            within_class_scatter = class_diff_from_mean.T @ class_diff_from_mean
            sep_data.append(class_data)
            sep_num_data.append(class_size)
            sep_mean_data.append(class_mean)
            sep_within_cls_sctr.append(within_class_scatter)
        ###########################################################################
        # Calculate scatter of the means
        ###########################################################################
        sep_mean_data = np.asarray(sep_mean_data)
        overall_mean = np.mean(sep_mean_data, axis=0)
        ###########################################################################
        # Calculate total within class scatter and between class scatter
        ###########################################################################
        s_within = np.sum(sep_within_cls_sctr, axis=0)
        sep_num_data = np.asarray(sep_num_data)
        cls_diff_frm_ov_mean = (sep_mean_data - overall_mean)
        s_between = ((cls_diff_frm_ov_mean.T) * sep_num_data) @ cls_diff_frm_ov_mean
        ###########################################################################
        # Find projection matrix
        ###########################################################################
        w = (np.linalg.pinv(s_within)) @ s_between
        self.w = w
        self.mean = sep_mean_data
        self.Sw = s_within
        self.Sb = s_between
        return w

    ###########################################################################
    # This function projects data according to dimension passed as argument
    ###########################################################################
    def projectData(self, dimension=1):
        ###########################################################################
        # Compute eigen-values and eigen-vectors of projection matrix and sort it
        ###########################################################################
        eps = 1e-8
        e_val, e_vec = np.linalg.eig(self.w)
        e_val[np.abs(e_val) < eps] = 0
        e_val_abs = np.absolute(e_val)
        e_sort_index = np.argsort(e_val_abs)[::-1]
        e_val = np.real(e_val[e_sort_index])
        e_vec = np.real(e_vec[:, e_sort_index])
        self.projection = e_vec[:, 0:dimension]
        return e_vec[:, 0:dimension], e_val[0:dimension]

    ###########################################################################
    # This function predicts predicts class labels of data samples based on
    # threshold passed as arguments. Currently supports 2 classes only
    ###########################################################################
    def predict(self, data, threshold):
        proj_mean = self.mean[0, :] @ self.projection
        if (proj_mean < threshold):
            lt_class = 0
            gt_class = 1
        else:
            lt_class = 1
            gt_class = 0
        prediction = np.zeros((data.shape[0],))
        prediction[np.argwhere(data < threshold)[:, 0]] = lt_class
        prediction[np.argwhere(data >= threshold)[:, 0]] = gt_class
        return prediction

