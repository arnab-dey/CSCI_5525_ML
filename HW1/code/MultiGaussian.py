#######################################################################
# IMPORTS
#######################################################################
import numpy as np
#######################################################################
# Class definitions
#######################################################################
class multiGaussian:
    #######################################################################
    # This function performs MLE
    #######################################################################
    def performMLE(self, data):
        if (None is data):
            return None, None
        ###########################################################################
        # Extract information from training data
        ###########################################################################
        N, D = data.shape
        if (0 == N):
            return None, None
        # Last column represents class label, therefore feature dimension is D-1
        D -= 1
        num_class = int(np.max(data[:, D])) + 1
        # Separate samples corresponding to each class
        estimated_priors = []
        estimated_mu = []
        estimated_sigma = []
        for classCount in range(num_class):
            class_data = data[data[:, D] == np.asarray(classCount)]
            class_size = np.size(class_data, 0)
            class_prior = class_size/N
            class_mean = np.mean(class_data[:, 0:D], axis=0)
            class_diff_from_mean = class_data[:, 0:D] - class_mean
            class_cov_matrix = class_diff_from_mean.T @ class_diff_from_mean/class_size
            # sep_data.append(class_data)
            # sep_num_data.append(class_size)
            estimated_priors.append(class_prior)
            estimated_mu.append(class_mean)
            estimated_sigma.append(class_cov_matrix)
        return np.asarray(estimated_priors), np.asarray(estimated_mu), np.asarray(estimated_sigma)

    ###########################################################################
    # This function implements the discriminant function logic and
    # classifies sample data according to the MLEs and priors given as input
    ###########################################################################
    def classify(self, data, prior, mu, sigma, isSharedSigma=False, isNaiveBayes=False):
        num_class = mu.shape[0]
        log_odds = np.zeros((data.shape[0], num_class))
        ###########################################################################
        # Discriminant function implementation
        ###########################################################################
        for classCount in range(num_class):
            class_mu = mu[classCount, :]
            if (False == isSharedSigma):
                class_sigma = sigma[classCount, :, :]
                log_sigma = np.linalg.det(class_sigma)
            else:
                class_sigma = sigma
            if (False == isNaiveBayes):
                quad_term = np.diagonal((data - class_mu) @ np.linalg.inv(class_sigma) @ (data - class_mu).T)
            else:
                s = np.diag(1./np.diag(class_sigma))
                quad_term = np.diagonal((data - class_mu) @ s @ (data - class_mu).T)
            if (False == isSharedSigma):
                g = -0.5 * log_sigma - 0.5 * quad_term + np.log(prior[classCount])
            else:
                g = - 0.5 * quad_term + np.log(prior[classCount])
            # Store generalized log odds for each class
            log_odds[:, classCount] = g
        return np.argmax(log_odds, axis=1)