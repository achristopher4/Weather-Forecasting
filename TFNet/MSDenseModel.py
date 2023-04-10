## Multi-Step Dense Model 
## Date: 4/5/2023

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

class MSDenseModel(tf.keras.Model):
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
                                                            mode='min')

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])

        return history