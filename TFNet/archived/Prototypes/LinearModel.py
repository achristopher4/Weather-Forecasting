## Base Linear Model 
## Date: 4/3/2023

import Window as w

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import IPython
import IPython.display
import matplotlib as mpl
import seaborn as sns

SEED = 44
MAX_EPOCHS = 20

class LinearModel(tf.keras.Model):
    def __init__(self, label_index = None) -> None:
        super().__init__()
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    
    def compile_and_fit(self, model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min',
                                                            restore_best_weights = True)

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val ) # ,
                            #callbacks=[early_stopping])
        
        ## Put plotting function: training and validation loss over epoch
        # Plot the training history and save the graph.
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.ylabel("MSE")
        plt.xlabel("Epoch")
        plt.title("Linear Training & Validation Loss")
        #plt.savefig("Train_History_Linear.png")
        plt.show()

        return history