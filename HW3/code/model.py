#######################################################################
# PACKAGE IMPORTS
#######################################################################
import tensorflow as tf
import numpy as np
#######################################################################
# MODEL DEFINITION
#######################################################################
class cnn_model:
    def __init__(self, max_epoch = 100, batch_size = 32, model_number=1):
        self.model = None
        self.opt = None
        self.loss_model = None
        self.accuracy_model = None
        self.batch_size = batch_size
        self.epochs = max_epoch
        self.cb = None
        self.verbose = 0
        self.model_number = model_number

    def model_build(self):
        if (1 == self.model_number):
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(20, kernel_size=(3, 3),
                                       padding='same', input_shape=(28, 28, 1)),  # Convolution layer
                tf.keras.layers.BatchNormalization(axis=-1),  # For normalization
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Max-pool 2x2
                tf.keras.layers.Dropout(0.5),  # Dropout layer with probability 0.5
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=tf.nn.relu, name='hidden_layer_1'),
                tf.keras.layers.Dropout(0.5),  # Dropout layer with probability 0.5
                tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output_layer')
            ])
        else:
            self.model = tf.keras.Sequential([
                tf.keras.Input((28, 28), name='mnist_feature'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=tf.nn.relu, name='hidden_layer_1'),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output_layer')
            ])

    def model_compile(self, opt, loss_model, accuracy_model):
        self.opt = opt
        self.loss_model = loss_model
        self.accuracy_model = accuracy_model
        self.model.compile(optimizer=opt,
                      loss=loss_model,
                      metrics=[accuracy_model])

    def model_fit(self, x, y, batch_size=None, epochs=None, cb=None, verbose=0):
        if (None is batch_size):
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size
        if (None is epochs):
            epochs = self.epochs
        else:
            self.epochs = epochs
        if (None is cb):
            cb = self.cb
        else:
            self.cb = cb
        self.model.fit(x, y, batch_size=batch_size,
                       epochs=epochs, callbacks=[cb], verbose=verbose)

    def model_update(self, opt):
        self.opt = opt
        self.model_compile(opt, self.loss_model, self.accuracy_model)

    def model_evaluate(self, x, y):
        loss, acc = self.model.evaluate(x, y, verbose=0)
        return loss, acc
