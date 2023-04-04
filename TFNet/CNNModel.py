## Testing CNN Model 
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

class CNNModel(tf.keras.Model):
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