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

    def fit(self, X, y):
        N_trn = X.shape[0]
        K = self.kernel(X, X, gamma=self.gamma)
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
        # m_idx = np.argwhere((a >= eps) & (self.C > a))[:, 0]
        # a_m = a[m_idx].reshape((m_idx.shape[0], 1))
        # y_m = y[m_idx].reshape((a_m.shape[0], 1))
        # X_m = X[m_idx, :]
        self.intercept = np.mean(self.s_vec_y[:, 0] - np.sum(K[sv_idx[:, None], sv_idx] * self.a[:, 0] * self.s_vec_y[:, 0], axis=1))

    def predict(self, X):
        K = self.kernel(X, self.s_vec)
        pred = np.sum(K * self.a[:, 0] * self.s_vec_y[:, 0], axis=1) + self.intercept
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred