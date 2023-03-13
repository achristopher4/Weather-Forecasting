## TFNet Model
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM


class Model():
    def __init__(self, train, test, cols, target) -> None:
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.train = train
        self.test = test
        self.pred_cols = cols
        self.target_level = 1000
        self.target_cols = target
        self.time_series()
    
    def time_series(self):
        date = self.train['time']
        inputData = self.train.loc[:, self.train.columns != 'time']
        model = Sequential(LSTM(250, input_shape = (date, inputData)))
        model.compile(loss='mae', optimizer='adam')