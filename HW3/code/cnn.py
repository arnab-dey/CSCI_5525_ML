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
is_plot_per_run_reqd = False
num_run = 10
num_classes = 10
batch_size = [32, 64, 96, 128]
max_epochs = 100 # max epoch
sgd_learning_rate = 0.01
sgd_momentum = 0.0  # not using momentum as asked in hw
adagrad_learning_rate = 0.1
adagrad_initial_accumulator_value = 0.01
adam_learning_rate = 0.001
adam_beta_1 = 0.9
adam_beta_2 = 0.999
loss_model = 'categorical_crossentropy'
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
def cnn():
    #######################################################################
    # Data loading
    #######################################################################
    mnist = tf.keras.datasets.mnist
    num_classes = 10
    num_channels = 1
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, num_channels)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, num_channels)
    x_train, x_test = x_train / 255.0, x_test / 255.0 # Scale data to [0, 1]
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    #######################################################################
    # Prepare callback
    #######################################################################
    cb = mycallback(patience=3)
    #######################################################################
    # (a) Batch size: 32, optimization: SGD
    #######################################################################
    opt = tf.keras.optimizers.SGD(learning_rate=sgd_learning_rate, momentum=sgd_momentum)
    #######################################################################
    # Build CNN model
    #######################################################################
    nn = cnn_model(model_number=1, max_epoch=max_epochs)
    nn.model_build()
    nn.model_compile(opt=opt, loss_model=loss_model, accuracy_model=accuracy_model)
    #######################################################################
    # Fit data
    #######################################################################
    print('###### Running part 1: SGD with batch size 32 ######')
    nn.model_fit(x_train, y_train, batch_size[0], max_epochs, cb, verbose=0)
    test_loss, test_accuracy = nn.model_evaluate(x_test, y_test)
    print('test loss = ', test_loss, ', test accuracy = ', test_accuracy)
    epoch_array = np.linspace(1, np.asarray(cb.loss_array).shape[0], num=np.asarray(cb.loss_array).shape[0])
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
            fig.savefig('./generatedPlots/q3_sgd_loss_acc.pdf')
        else:
            plt.show()
    #######################################################################
    # (b) Different optimization model, different batch size
    #######################################################################
    sgd_train_times = np.zeros((num_run, len(batch_size)))
    sgd_loss_array = []
    sgd_acc_array = []
    adagrad_train_times = np.zeros((num_run, len(batch_size)))
    adagrad_loss_array = []
    adagrad_acc_array = []
    adam_train_times = np.zeros((num_run, len(batch_size)))
    adam_loss_array = []
    adam_acc_array = []
    # SGD optimizer
    opt_sgd = tf.keras.optimizers.SGD(learning_rate=sgd_learning_rate, momentum=sgd_momentum)
    cnn_sgd = cnn_model(model_number=1, max_epoch=max_epochs)
    cnn_sgd.model_build()
    cnn_sgd.model_compile(opt=opt_sgd, loss_model=loss_model, accuracy_model=accuracy_model)
    cb_sgd = mycallback(patience=3)
    cnn_sgd.cb = cb_sgd
    # Adagrad optimizer
    opt_adagrad = tf.keras.optimizers.Adagrad(learning_rate=adagrad_learning_rate,
                                              initial_accumulator_value=adagrad_initial_accumulator_value,
                                              epsilon=1e-5)
    cnn_adagrad = cnn_model(model_number=1, max_epoch=max_epochs)
    cnn_adagrad.model_build()
    cnn_adagrad.model_compile(opt=opt_adagrad, loss_model=loss_model, accuracy_model=accuracy_model)
    cb_adagrad = mycallback(patience=3)
    cnn_adagrad.cb = cb_adagrad
    # Adam optimizer
    opt_adam = tf.keras.optimizers.Adam(learning_rate=adam_learning_rate, beta_1=adam_beta_1,
                                        beta_2=adam_beta_2, epsilon=1e-5)
    cnn_adam = cnn_model(model_number=1, max_epoch=max_epochs)
    cnn_adam.model_build()
    cnn_adam.model_compile(opt=opt_adam, loss_model=loss_model, accuracy_model=accuracy_model)
    cb_adam = mycallback(patience=3)
    cnn_adam.cb = cb_adam
    for run_idx in range(num_run):
        print('###### RUN NUMBER ', run_idx, ' ######')
        # Iterate over different batch sizes
        for batch_idx in range(len(batch_size)):
            #######################################################################
            # SGD
            #######################################################################
            print('###### Running part 2: SGD with batch size ', batch_size[batch_idx], ' ######')
            cnn_sgd.model_fit(x_train, y_train, batch_size[batch_idx])
            time_taken = cb_sgd.time_array[0]
            sgd_train_times[run_idx, batch_idx] = time_taken[1]
            print('SGD: run no. = ', run_idx, ', batch size = ', batch_size[batch_idx], ' convergence time = ', time_taken[1])
            test_loss, test_accuracy = cnn_sgd.model_evaluate(x_test, y_test)
            print('SGD: run no. = ', run_idx, ', batch size = ', batch_size[batch_idx],
                  ', test loss = ', test_loss, ', test accuracy = ', test_accuracy)
            sgd_loss_array.append(np.asarray(cb_sgd.loss_array))
            sgd_acc_array.append(np.asarray(cb_sgd.acc_array))
            epoch_array_sgd = np.linspace(1, np.asarray(cb_sgd.loss_array).shape[0],
                                          num=np.asarray(cb_sgd.loss_array).shape[0])
            #######################################################################
            # ADAGRAD
            #######################################################################
            print('###### Running part 2: ADAGRAD with batch size ', batch_size[batch_idx], ' ######')
            cnn_adagrad.model_fit(x_train, y_train, batch_size[batch_idx])
            time_taken = cb_adagrad.time_array[0]
            adagrad_train_times[run_idx, batch_idx] = time_taken[1]
            print('ADAGRAD: run no. = ', run_idx, ', batch size = ', batch_size[batch_idx], ' convergence time = ', time_taken[1])
            test_loss, test_accuracy = cnn_adagrad.model_evaluate(x_test, y_test)
            print('ADAGRAD: run no. = ', run_idx, ', batch size = ', batch_size[batch_idx],
                  ', test loss = ', test_loss, ', test accuracy = ', test_accuracy)
            adagrad_loss_array.append(np.asarray(cb_adagrad.loss_array))
            adagrad_acc_array.append(np.asarray(cb_adagrad.acc_array))
            epoch_array_adagrad = np.linspace(1, np.asarray(cb_adagrad.loss_array).shape[0],
                                              num=np.asarray(cb_adagrad.loss_array).shape[0])
            #######################################################################
            # ADAM
            #######################################################################
            print('###### Running part 2: ADAM with batch size ', batch_size[batch_idx], ' ######')
            cnn_adam.model_fit(x_train, y_train, batch_size[batch_idx])
            time_taken = cb_adam.time_array[0]
            adam_train_times[run_idx, batch_idx] = time_taken[1]
            print('ADAM: run no. = ', run_idx, ', batch size = ', batch_size[batch_idx], ' convergence time = ', time_taken[1])
            test_loss, test_accuracy = cnn_adam.model_evaluate(x_test, y_test)
            print('ADAM: run no. = ', run_idx, ', batch size = ', batch_size[batch_idx],
                  ', test loss = ', test_loss, ', test accuracy = ', test_accuracy)
            adam_loss_array.append(np.asarray(cb_adam.loss_array))
            adam_acc_array.append(np.asarray(cb_adam.acc_array))
            epoch_array_adam = np.linspace(1, np.asarray(cb_adam.loss_array).shape[0],
                                           num=np.asarray(cb_adam.loss_array).shape[0])
            #######################################################################
            # Dataplots
            #######################################################################
            if (True == is_plot_per_run_reqd):
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
                    ax.plot(epoch_array_sgd, np.asarray(cb_sgd.loss_array), label='SGD loss')
                    ax.plot(epoch_array_adagrad, np.asarray(cb_adagrad.loss_array), label='ADAGRAD loss')
                    ax.plot(epoch_array_adam, np.asarray(cb_adam.loss_array), label='ADAM loss')
                    # ax.plot(epoch_array, np.asarray(cb.acc_array), label='accuracy')

                    ax.set_xlabel(r'number of epochs', fontsize=8)
                    ax.set_ylabel(r'loss', fontsize=8)

                    plt.legend()
                    if (True == isPlotPdf):
                        if not os.path.exists('./generatedPlots'):
                            os.makedirs('generatedPlots')
                        fig.savefig('./generatedPlots/q3_loss_batch_' + str(batch_size[batch_idx])
                                    + '_run_' + str(run_idx) + '.pdf')
                    else:
                        plt.show()
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
                    ax.plot(epoch_array_sgd, np.asarray(cb_sgd.acc_array), label='SGD accuracy')
                    ax.plot(epoch_array_adagrad, np.asarray(cb_adagrad.acc_array), label='ADAGRAD accuracy')
                    ax.plot(epoch_array_adam, np.asarray(cb_adam.acc_array), label='ADAM accuracy')

                    ax.set_xlabel(r'number of epochs', fontsize=8)
                    ax.set_ylabel(r'accuracy', fontsize=8)

                    plt.legend()
                    if (True == isPlotPdf):
                        if not os.path.exists('./generatedPlots'):
                            os.makedirs('generatedPlots')
                        fig.savefig('./generatedPlots/q3_acc_batch_' + str(batch_size[batch_idx])
                                    + '_run_' + str(run_idx) + '.pdf')
                    else:
                        plt.show()

    #######################################################################
    # Plot of convergence time: SGD
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
        ax.plot(batch_size, np.mean(sgd_train_times, axis=0), label='SGD convergence time')

        ax.set_xlabel(r'batch size', fontsize=8)
        ax.set_ylabel(r'convergence time (s)', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q3_sgd_conv_time.pdf')
        else:
            plt.show()

    #######################################################################
    # Plot of convergence time: ADAGRAD
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
        ax.plot(batch_size, np.mean(adagrad_train_times, axis=0), label='ADAGRAD convergence time')

        ax.set_xlabel(r'batch size', fontsize=8)
        ax.set_ylabel(r'convergence time (s)', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q3_adagrad_conv_time.pdf')
        else:
            plt.show()

    #######################################################################
    # Plot of convergence time: ADAM
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
        ax.plot(batch_size, np.mean(adam_train_times, axis=0), label='ADAM convergence time')

        ax.set_xlabel(r'batch size', fontsize=8)
        ax.set_ylabel(r'convergence time (s)', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q3_adam_conv_time.pdf')
        else:
            plt.show()

    #######################################################################
    # Plot of convergence time: ALL
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
        ax.plot(batch_size, np.mean(sgd_train_times, axis=0), label='SGD convergence time')
        ax.plot(batch_size, np.mean(adagrad_train_times, axis=0), label='ADAGRAD convergence time')
        ax.plot(batch_size, np.mean(adam_train_times, axis=0), label='ADAM convergence time')

        ax.set_xlabel(r'batch size', fontsize=8)
        ax.set_ylabel(r'convergence time (s)', fontsize=8)

        plt.legend()
        if (True == isPlotPdf):
            if not os.path.exists('./generatedPlots'):
                os.makedirs('generatedPlots')
            fig.savefig('./generatedPlots/q3_all_conv_time.pdf')
        else:
            plt.show()