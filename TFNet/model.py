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
        self.base_model()
    
    def base_model(self):
        ## Time Series

        trainIndex = self.train[['lat', 'lon', 'time']]
        trainInputData = self.train[self.train.columns.difference(['lat', 'lon', 'time', 't2m', 'tcc', 'tp'])]
        trainAns = self.train[['lat', 'lon', 'time', 't2m', 'tcc', 'tp']]

        """model = Sequential(LSTM(250, input_shape = (trainDate, trainInputData)))
        testDate = self.test[['lat', 'lon', 'time']]
        testInputData = self.test[self.test.columns.difference(['lat', 'lon', 'time'])]
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        history = model.fit(trainDate, trainInputData, epochs=250, batch_size=72, validation_data=(testDate, testInputData), verbose=2, shuffle=False)

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()"""