#######################################################################
# PACKAGE IMPORTS
#######################################################################
import numpy as np
import data_processor as dp
#######################################################################
# Class definitions
#######################################################################
class stump:
    def __init__(self, feature_map=None):
        self.root_feature = None
        self.pred_label = None
        self.factor_max_lim = 5.
        self.split_points = None
        self.root_factors = None
        self.feature_map = feature_map

    #######################################################################
    # This function calculates entropy of a random variable
    #######################################################################
    def calculate_entropy(self, y_labels, D=None):
        # Get total number of samples
        n_sample = y_labels.shape[0]
        # Get number of labels
        labels = np.unique(y_labels)
        entropy = 0.
        for label in labels:
            sample_idx = y_labels[:] == label
            if (None is D):
                p_i = np.sum(sample_idx)/n_sample
            else:
                # Calculate probability based on sample weights
                p_i = np.sum(D[sample_idx])/np.sum(D)
            # Avoid log(0)
            if (p_i == 0.):
                log_p_i = -1e+8 # Setting to a large value to avoid log 0
            else:
                log_p_i = np.log2(p_i)
            entropy -= p_i*log_p_i
        return entropy

    #######################################################################
    # Training function for the stump
    #######################################################################
    def train(self, X, y, sample_weights=None):
        n_sample = X.shape[0]
        n_features = X.shape[1]
        n_labels = np.unique(y).shape[0]
        #######################################################################
        # Calculate entropy of the root node
        #######################################################################
        root_entropy = self.calculate_entropy(y, sample_weights)
        information_gain = np.zeros((n_features,))
        self.split_points = np.zeros((n_features,))
        majority_labels = []
        #######################################################################
        # Find the best feature to split
        #######################################################################
        for feature in range(n_features):
            factors = np.unique(X[:, feature])
            #######################################################################
            # Checking for categorical variable
            #######################################################################
            if (factors.shape[0] > self.factor_max_lim):
                self.split_points[feature] = np.mean(X[:, feature])
                feature_val = dp.convert_to_categorical(X[:, feature], self.split_points[feature])
                factors = np.array([0., 1.])
            else:
                self.split_points[feature] = np.inf
                feature_val = X[:, feature]

            fac_majority_label = np.zeros((factors.shape[0],))
            cond_entropy = 0.
            #######################################################################
            # Calculate entropy for each leaf node for a particular feature
            #######################################################################
            for factor_idx in range(factors.shape[0]):
                factor = factors[factor_idx]
                if (None is sample_weights):
                    p_value = np.sum(feature_val == factor)/n_sample # This is applicable only with same sample weights
                    cond_entropy += p_value * (self.calculate_entropy(y[feature_val == factor]))
                else:
                    p_value = np.sum(sample_weights[feature_val == factor])/np.sum(sample_weights)
                    cond_entropy += p_value*(self.calculate_entropy(y[feature_val == factor], sample_weights[feature_val == factor]))
                #######################################################################
                # Find majority vote for samples in a leaf node: This will be
                # the prediction label for this node
                #######################################################################
                (values, counts) = np.unique(y[feature_val == factor], return_counts=True)
                majority_label = values[np.argmax(counts)]
                fac_majority_label[factor_idx] = majority_label

            majority_labels.append(fac_majority_label)
            #######################################################################
            # Find information gain for a particular feature
            #######################################################################
            information_gain[feature] = root_entropy-cond_entropy
        #######################################################################
        # Find the best feature from the information gain values
        #######################################################################
        self.root_feature = int(np.argmax(information_gain))
        self.pred_label = majority_labels[int(self.root_feature)]
        #######################################################################
        # Stores info about leaf nodes split conditions
        #######################################################################
        if (self.split_points[self.root_feature] != np.inf):
            self.root_factors = np.unique(dp.convert_to_categorical(X[:, self.root_feature], self.split_points[self.root_feature]))
        else:
            self.root_factors = np.unique(X[:, self.root_feature])

    #######################################################################
    # Prediction function
    #######################################################################
    def predict(self, X_test, is_feat_map_reqd=False):
        n_test = X_test.shape[0]
        pred = np.zeros((X_test.shape[0]))
        #######################################################################
        # Feature map will be useful in case of random forest
        # Maps internal feature index to actual feature index in the data
        #######################################################################
        if (True == is_feat_map_reqd):
            if (None is self.feature_map):
                root_feature = self.root_feature
            else:
                root_feature = self.feature_map[self.root_feature]
        else:
            root_feature = self.root_feature
        #######################################################################
        # Check for continuous valued feature
        #######################################################################
        if (self.split_points[self.root_feature] != np.inf):
            # X_test[:, self.root_feature] = dp.convert_to_categorical(X_test[:, self.root_feature], self.split_points[self.root_feature])
            feature_val = dp.convert_to_categorical(X_test[:, root_feature], self.split_points[self.root_feature])
        else:
            feature_val = X_test[:, root_feature]
        #######################################################################
        # Predict each sample in the test dataset
        #######################################################################
        for sample in range(n_test):
            sample_factor = feature_val[sample]
            if (sample_factor in self.root_factors):
                factor_idx = int(np.argwhere(self.root_factors[:] == sample_factor))
                pred[sample] = self.pred_label[factor_idx]
            else:
                pred[sample] = np.random.choice(self.pred_label)
        return pred
