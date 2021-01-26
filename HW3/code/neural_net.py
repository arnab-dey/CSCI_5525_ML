#######################################################################
# PACKAGE IMPORTS
#######################################################################
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from model import cnn_model
from utils import mycallback
################################################################################
# Variable declaration
################################################################################
isPlotReqd = True
isPlotPdf = True
num_classes = 10
batch_size = 32
max_epochs = 100 # max epoch
sgd_learning_rate = 0.01
sgd_momentum = 0.0  # Not using momentum as asked in the hw
# loss_model = 'sparse_categorical_crossentropy'
loss_model = 'categorical_crossentropy'
# accuracy_model = 'sparse_categorical_accuracy'
accuracy_model = 'accuracy'
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
# MODEL DEFINITION
#######################################################################
def neural_net():
    #######################################################################
    # Data loading
    #######################################################################
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Scale data to [0, 1]
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    #######################################################################
    # Prepare callback
    #######################################################################
    cb = mycallback(patience=3)
    #######################################################################
    # Build feed forward model
    #######################################################################
    opt = tf.keras.optimizers.SGD(learning_rate=sgd_learning_rate, momentum=sgd_momentum)
    nn = cnn_model(model_number=0)
    nn.model_build()
    nn.model_compile(opt=opt, loss_model=loss_model, accuracy_model=accuracy_model)
    #######################################################################
    # Fit data
    #######################################################################
    nn.model_fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, cb=cb, verbose=0)
    epoch_array = np.linspace(1, np.asarray(cb.loss_array).shape[0], num=np.asarray(cb.loss_array).shape[0])
    test_loss, test_accuracy = nn.model_evaluate(x_test, y_test)
    print('test loss = ', test_loss, ', test accuracy = ', test_accuracy)
    #######################################################################
    # Plot of loss curve
    #######################################################################
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
        ax.plot(epoch_array, np.asarray(cb.loss_array), label='loss')
        ax.plot(epoch_array, np.asarray(cb.acc_array), label='accuracy')

        ax.set_xlabel(r'number of epochs', fontsize=8)
        ax.set_ylabel(r'metrics', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q2_loss_acc.pdf')
        else:
            plt.show()