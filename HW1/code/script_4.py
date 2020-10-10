#######################################################################
# IMPORTS
#######################################################################
import numpy as np
import os
import logisticRegression as lr
import naiveBayesGaussian as nbgauss
import matplotlib.pyplot as plt
import matplotlib as mpl
################################################################################
# Variable declaration
################################################################################
isPlotReqd = True
isPlotPdf = True
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
# CODE STARTS HERE
#######################################################################
train_percent = [10, 25, 50, 75, 100]
num_split = 10
logReg = lr.discriminator()
naiveBayes = nbgauss.naiveBayes()
#######################################################################
# Load Boston50 data
#######################################################################
logReg.loadData('Boston50')
naiveBayes.loadData('Boston50')
print('Running LR and GNB for Boston50')
et_lr_boston50, ev_lr_boston50 = logReg.LogisticRegression(num_split, train_percent)
et_gnb_boston50, ev_gnb_boston50 = naiveBayes.naiveBayesGaussian(num_split, train_percent)
#######################################################################
# Load Boston75 data
#######################################################################
logReg.loadData('Boston75')
naiveBayes.loadData('Boston75')
print('Running LR and GNB for Boston75')
et_lr_boston75, ev_lr_boston75 = logReg.LogisticRegression(num_split, train_percent)
et_gnb_boston75, ev_gnb_boston75 = naiveBayes.naiveBayesGaussian(num_split, train_percent)
#######################################################################
# Load digits data
#######################################################################
logReg.loadData('digits')
naiveBayes.loadData('digits')
print('Running LR and GNB for digits')
et_lr_digits, ev_lr_digits = logReg.LogisticRegression(num_split, train_percent)
et_gnb_digits, ev_gnb_digits = naiveBayes.naiveBayesGaussian(num_split, train_percent)
#######################################################################
# Plot of error curves
#######################################################################
if (True == isPlotReqd):
    #######################################################################
    # Dataset: Boston50
    #######################################################################
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
    ax.errorbar(train_percent, np.mean(ev_lr_boston50, axis=0), np.std(ev_lr_boston50, axis=0), label='LR')
    ax.errorbar(train_percent, np.mean(ev_gnb_boston50, axis=0), np.std(ev_gnb_boston50, axis=0), label='GNB')

    ax.set_xlabel(r'Training data used (\%)', fontsize=8)
    ax.set_ylabel(r'error (\%)', fontsize=8)

    plt.legend()
    if (True == isPlotPdf):
        if not os.path.exists('./generatedPlots'):
            os.makedirs('generatedPlots')
        fig.savefig('./generatedPlots/Q4_boston50_test_err.pdf')
    else:
        plt.show()

    #######################################################################
    # Dataset: Boston75
    #######################################################################
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
    ax.errorbar(train_percent, np.mean(ev_lr_boston75, axis=0), np.std(ev_lr_boston75, axis=0), label='LR')
    ax.errorbar(train_percent, np.mean(ev_gnb_boston75, axis=0), np.std(ev_gnb_boston75, axis=0), label='GNB')

    ax.set_xlabel(r'Training data used (\%)', fontsize=8)
    ax.set_ylabel(r'error (\%)', fontsize=8)

    plt.legend()
    if (True == isPlotPdf):
        if not os.path.exists('./generatedPlots'):
            os.makedirs('generatedPlots')
        fig.savefig('./generatedPlots/Q4_boston75_test_err.pdf')
    else:
        plt.show()

    #######################################################################
    # Dataset: Digits
    #######################################################################
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
    ax.errorbar(train_percent, np.mean(ev_lr_digits, axis=0), np.std(ev_lr_digits, axis=0), label='LR')
    ax.errorbar(train_percent, np.mean(ev_gnb_digits, axis=0), np.std(ev_gnb_digits, axis=0), label='GNB')

    ax.set_xlabel(r'Training data used (\%)', fontsize=8)
    ax.set_ylabel(r'error (\%)', fontsize=8)

    plt.legend()
    if (True == isPlotPdf):
        if not os.path.exists('./generatedPlots'):
            os.makedirs('generatedPlots')
        fig.savefig('./generatedPlots/Q4_digits_test_err.pdf')
    else:
        plt.show()