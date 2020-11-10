#######################################################################
# PACKAGE IMPORTS
#######################################################################
import tensorflow as tf
import numpy as np
import time
#######################################################################
# CALLBACK IMPLEMENTATION
#######################################################################
class mycallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, loss_model='categorical_crossentropy', accuracy_model='accuracy'):
        super(mycallback, self).__init__()
        self.patience = patience # No. of consecutive epochs to wait before we declare convergence
        self.loss_array = []
        self.acc_array = []
        self.time_array = []
        self.train_start_time = None
        self.loss_model = loss_model
        self.accuracy_model = accuracy_model

    #######################################################################
    # Callback evoked at start of training
    #######################################################################
    def on_train_begin(self, logs=None):
        self.wait_count_before_conv = 0
        self.wait_count_loss_inc = 0
        self.prev_loss = np.Inf
        self.epoch_num = 0
        self.loss_eps = 0.001
        self.train_start_time = time.time()
        self.loss_array = []
        self.acc_array = []
        self.time_array = []

    #######################################################################
    # Callback evoked at each epoch end
    # Two criteria for convergence:
    # 1. Loss increases for self.patience number of consecutive epochs
    # 2. Loss reduction is less than self.loss_eps for self.patience
    # number of consecutive epochs
    #######################################################################
    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        accuracy = logs[self.accuracy_model]
        print('Epoch: ', epoch+1, ', loss = ', loss, ', accuracy = ', accuracy)
        self.epoch_num = epoch
        if (loss < self.prev_loss):
            self.best_weights = self.model.get_weights()
            self.loss_array.append(logs["loss"])
            self.acc_array.append(logs[self.accuracy_model])
            self.wait_count_loss_inc = 0
            if (np.abs(loss-self.prev_loss) <= self.loss_eps):
                self.wait_count_before_conv += 1
            else:
                self.wait_count_before_conv = 0

            self.prev_loss = loss
        else:
            self.wait_count_loss_inc += 1
        if ((self.wait_count_loss_inc >= self.patience)
                or (self.wait_count_before_conv >= self.patience)):
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        self.time_array.append((self.epoch_num, time.time()-self.train_start_time))
