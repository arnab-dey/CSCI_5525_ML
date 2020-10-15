#######################################################################
# IMPORTS
#######################################################################
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.svm import SVC
#######################################################################
# Function definitions
#######################################################################
def linear_kernel(x,y, gamma=None):
    return x @ y.T
def rbf_kernel(x, y, gamma=2.0):
    N_x = x.shape[0]
    N_y = y.shape[0]
    K = []
    for index in range(N_y):
        K_index = np.exp(-gamma * np.linalg.norm(x - y[index, :], axis=1)**2)
        K.append(K_index)
    K = np.asarray(K)
    return K.T
#######################################################################
# Class definitions
#######################################################################
class svm():
    def __init__(self, kernel='linear', C=0., gamma=None):
        if (kernel == 'linear'):
            self.kernel = linear_kernel
        elif (kernel == 'rbf'):
            self.kernel = rbf_kernel
        else:
            self.kernel = rbf_kernel
        self.C = C
        self.gamma = gamma
        self.a = None
        self.s_vec = None
        self.s_vec_y = None
        self.coef = None
        self.intercept = None
        self.num_class = None

    def fit(self, X, y):
        N_trn = X.shape[0]
        K = self.kernel(X, X, gamma=self.gamma)
        # Check if multi-class
        self.num_class = int(np.max(y)) + 1
        if (self.num_class > 2):
            self.a = []
            self.s_vec = []
            self.s_vec_y = []
            self.intercept = []
            if (self.kernel == linear_kernel):
                self.coef = []
            else:
                self.coef = None
            y_k = np.zeros((y.shape[0],))
            for class_idx in range(self.num_class):
                # Process labels for current class
                y_k[:] = y
                y_k[y_k != class_idx] = -1
                y_k[y_k == class_idx] = 1
                # Prepare optimization problem
                H = np.outer(y_k, y_k) * K
                cvxopt_solvers.options['show_progress'] = False
                P = cvxopt_matrix(H)
                q = cvxopt_matrix(-np.ones((N_trn, 1)))
                G = cvxopt_matrix(np.vstack((np.eye(N_trn) * -1, np.eye(N_trn))))
                h = cvxopt_matrix(np.hstack((np.zeros(N_trn), np.ones(N_trn) * self.C)))
                A = cvxopt_matrix(y_k.reshape(1, -1))
                b = cvxopt_matrix(np.zeros(1))

                # Solve optimization problem
                sol = cvxopt_solvers.qp(P, q, G, h, A, b)
                # Get the lagrange multipliers
                a = np.array(sol['x'])
                # Find the support vectors and store them
                eps = 1e-6
                sv_idx = np.argwhere(a >= eps)[:, 0]
                self.a.append(a[sv_idx].reshape((sv_idx.shape[0], 1)))
                self.s_vec.append(X[sv_idx, :])
                self.s_vec_y.append(y_k[sv_idx].reshape((sv_idx.shape[0], 1)))
                if (self.kernel == linear_kernel):
                    self.coef.append(((y_k.reshape((y_k.shape[0], 1)) * a).T @ X).T)
                else:
                    self.coef = None
                self.intercept.append(np.mean(self.s_vec_y[class_idx][:, 0] - np.sum(
                    K[sv_idx[:, None], sv_idx] * self.a[class_idx][:, 0] * self.s_vec_y[class_idx][:, 0], axis=1)))
        else:
            H = np.outer(y, y) * K
            cvxopt_solvers.options['show_progress'] = False
            P = cvxopt_matrix(H)
            q = cvxopt_matrix(-np.ones((N_trn, 1)))
            G = cvxopt_matrix(np.vstack((np.eye(N_trn) * -1, np.eye(N_trn))))
            h = cvxopt_matrix(np.hstack((np.zeros(N_trn), np.ones(N_trn) * self.C)))
            A = cvxopt_matrix(y.reshape(1, -1))
            b = cvxopt_matrix(np.zeros(1))

            # Solve optimization problem
            sol = cvxopt_solvers.qp(P, q, G, h, A, b)
            # Get the lagrange multipliers
            a = np.array(sol['x'])
            # Find the support vectors and store them
            eps = 1e-6
            sv_idx = np.argwhere(a >= eps)[:, 0]
            self.a = a[sv_idx].reshape((sv_idx.shape[0], 1))
            self.s_vec = X[sv_idx, :]
            self.s_vec_y = y[sv_idx].reshape((sv_idx.shape[0], 1))
            # Calculate weights and intercept
            if (self.kernel == linear_kernel):
                    self.coef = ((y.reshape((y.shape[0], 1)) * a).T @ X).T
            else:
                self.coef = None
            self.intercept = np.mean(self.s_vec_y[:, 0] - np.sum(K[sv_idx[:, None], sv_idx] * self.a[:, 0] * self.s_vec_y[:, 0], axis=1))

    def predict(self, X):
        if (self.num_class > 2):
            pred = np.zeros((X.shape[0], self.num_class))
            for class_idx in range(self.num_class):
                K = self.kernel(X, self.s_vec[class_idx])
                pred[:, class_idx] = np.sum(K * self.a[class_idx][:, 0] * self.s_vec_y[class_idx][:, 0], axis=1) + self.intercept[class_idx]
            return np.argmax(pred, axis=1)
        else:
            K = self.kernel(X, self.s_vec)
            pred = np.sum(K * self.a[:, 0] * self.s_vec_y[:, 0], axis=1) + self.intercept
            pred[pred >= 0] = 1
            pred[pred < 0] = -1
            return pred