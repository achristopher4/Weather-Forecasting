## TFNet Model
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


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
        trainDate = self.train['time']
        trainInputData = self.train.loc[:, self.train.columns != 'time']
        model = Sequential(LSTM(250, input_shape = (trainDate, trainInputData)))
        testDate = self.test['time']
        testInputData = self.test.loc[:, self.train.columns != 'time']
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        history = model.fit(trainDate, trainInputData, epochs=250, batch_size=72, validation_data=(testDate, testInputData), verbose=2, shuffle=False)

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()