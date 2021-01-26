#######################################################################
# PACKAGE IMPORTS
#######################################################################
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import io
from scipy.spatial import distance as scp_sp_dist
#######################################################################
# Static variable declaration
#######################################################################
isPlotReqd = True
isPlotPdf = True
isImageSaved = True
train_percent = 0.8
################################################################################
# Settings for plot
################################################################################
if (True == isPlotReqd):
    if (True == isPlotPdf):
        mpl.use('pdf')
        fig_width  = 3.487
        fig_height = fig_width / 1.618
        rcParams = {
            'font.family': 'serif',
            'font.serif': 'Times',
            'text.usetex': True,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.labelsize': 8,
            'legend.fontsize': 8,
            'figure.figsize': [fig_width, fig_height]
           }
        plt.rcParams.update(rcParams)
#######################################################################
# Function definitions
#######################################################################
class my_kmeans:
    def __init__(self, k, n_init=10):
        self.n_clusters = k
        # self.mu = None
        self.loss_bound = 1e-4
        self.n_initialization = int(n_init)
        self.cluster_centers_ = None
        self.labels_ = None
        self.best_loss_ = None
        self.best_iter_ = None
    def fit(self, X):
        ###########################################################################
        # Get number of samples and features
        ###########################################################################
        n_samples = X.shape[0]
        n_features = X.shape[1]
        loss_init_arr = np.zeros((self.n_initialization,))
        mu_init_arr = []
        loss_storage = []
        labels_arr = []
        itr_arr = []
        for init_idx in range(self.n_initialization):
            ###########################################################################
            # Initialize means at random values
            ###########################################################################
            np.random.seed(init_idx*10)
            mu = np.random.rand(int(self.n_clusters), n_features)
            loss_change = 1000. # Initialization
            prev_loss = 0.
            loss = 1000.
            loss_arr = []
            itr = 0
            while (loss_change >= self.loss_bound):
                itr += 1
                loss = 0. # Initialization
                ###########################################################################
                # Calculate distance from cluster centers and predict labels
                ###########################################################################
                cdist = scp_sp_dist.cdist(mu, X, 'euclidean')
                ###########################################################################
                # Assign class to each sample based on lowest distance to cluster means
                ###########################################################################
                kmeans_index = (np.argpartition(cdist, 1, axis=0))[0, :]
                ###########################################################################
                # Update cluster centers
                ###########################################################################
                for mu_idx in range(mu.shape[0]):
                    samp_in_clus_idx = kmeans_index == mu_idx
                    if (samp_in_clus_idx.any() == True):
                        mu[mu_idx, :] = np.mean(X[samp_in_clus_idx, :], axis=0)
                        ###########################################################################
                        # Compute loss
                        ###########################################################################
                        loss += np.sum(scp_sp_dist.cdist(mu[mu_idx, :].reshape((1, mu.shape[1])), X[samp_in_clus_idx, :], 'euclidean'))

                ###########################################################################
                # Store the total loss value
                ###########################################################################
                loss_arr.append(loss)
                loss_change = np.abs(prev_loss - loss)
                prev_loss = loss
                ###########################################################################
                # Console log
                ###########################################################################
                print('KMeans: K = ', self.n_clusters, ' Run' , init_idx, ' Iteration = ', itr, ' loss = ', loss)
            ###########################################################################
            # Store final loss, whole loss array, final mu, labels
            ###########################################################################
            itr_arr.append(itr)
            loss_init_arr[init_idx] = loss
            mu_init_arr.append(mu)
            loss_storage.append(np.asarray(loss_arr))
            labels_arr.append(np.asarray(kmeans_index))
        ###########################################################################
        # Find the best value over all runs
        ###########################################################################
        best_run_idx = int(np.argmin(loss_init_arr))
        print('Lowest loss achieved in run ', best_run_idx)
        self.best_loss_ = loss_storage[best_run_idx]
        self.cluster_centers_ = mu_init_arr[best_run_idx]
        self.labels_ = labels_arr[best_run_idx]
        self.best_iter_ = itr_arr[best_run_idx]
        ###########################################################################
        # Plot the loss
        ###########################################################################
        if (True == isPlotReqd):
            ###########################################################################
            # Configure axis and grid
            ###########################################################################
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

            ax.set_axisbelow(True)
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', linewidth='0.5')
            ax.grid(which='minor', linestyle="-.", linewidth='0.5')

            x_axis = np.linspace(1, len(self.best_loss_), num=int(len(self.best_loss_)))
            ax.plot(x_axis, self.best_loss_, label='Loss')

            ax.set_xlabel(r'number of iterations', fontsize=8)
            ax.set_ylabel(r'Loss', fontsize=8)

            plt.legend()
            if (True == isPlotPdf):
                if not os.path.exists('./generatedPlots'):
                    os.makedirs('generatedPlots')
                filename = 'q3_loss_'+str(self.n_clusters)+'.pdf'
                fig.savefig('./generatedPlots/'+filename)
            else:
                plt.show()


def kmeans(image):
    ###########################################################################
    # Check if dataset is present in the location
    ###########################################################################
    if not os.path.isfile(image):
        print("data data file can't be located")
        exit(1)
    ###########################################################################
    # Proceed with k-means
    ###########################################################################
    img = io.imread(image)
    img = img / 255 # Normalizing all values to 0-1
    ###########################################################################
    # Reshaping image to represent RGB values column wise
    ###########################################################################
    height, width, col_depth = img.shape
    img = np.reshape(img, ((height * width), col_depth))
    N = (img.shape)[0]  # number of samples
    D = (img.shape)[1]  # feature dimension
    ###########################################################################
    # Parameters for kmeans
    ###########################################################################
    k_arr = [3, 5, 7]
    ###########################################################################
    # Run KMeans
    ###########################################################################
    for k_idx in range(len(k_arr)):
        k = k_arr[k_idx]
        ###########################################################################
        # Initialize means at random values
        ###########################################################################
        km = my_kmeans(k)
        km.fit(img)
        ###########################################################################
        # Get the values of the cluster centers
        ###########################################################################
        m_i = km.cluster_centers_
        ###########################################################################
        # Get the labels of the pixels
        ###########################################################################
        k_means_labels = km.labels_
        ###########################################################################
        # Reconstruct image
        ###########################################################################
        compressed_image = []
        for pixel in range(N):
            color = k_means_labels[pixel]
            compressed_image.append(m_i[color, :])
        ###########################################################################
        # Save or show image
        ###########################################################################
        compressed_image = 255.*np.reshape(compressed_image, (height, width, col_depth))
        fname = 'compressed_img_' + str(k)
        if (True == isImageSaved):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            io.imsave('./generatedPlots/'+fname+'.png', compressed_image)
        else:
            io.imshow(compressed_image)
            io.show()


