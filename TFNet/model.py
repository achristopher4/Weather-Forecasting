## Final Iteration of TFNet Model
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager

import window as w

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import IPython
import IPython.display
import matplotlib as mpl
import seaborn as sns

SEED = 44

class Model(tf.keras.Model):
    def __init__(self, train, validation, test, label_index = None) -> None:
        super().__init__()
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        self.train_data = train
        self.validation_data = validation
        self.test_data = test
        self.label_index = label_index

    def singleStep(self):
        print(f"\nSingle Step Model")
        single_step_window = w.WindowGenerator(input_width= 1, label_width= 1, 
                                               shift= 1, train_df= self.train_data, 
                                               val_df= self.validation_data,  
                                               test_df= self.test_data,
                                               label_columns= ['t'])
        print(f"{single_step_window}\n")

        for example_inputs, example_labels in single_step_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')
        print()


    def linearModel(self):
        pass

    def denseModel(self):
        pass

    def multiStepDense(self):
        pass
    
    def baseCNN(self, CONV_WIDTH):
        conv_model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=32,
                        kernel_size=(CONV_WIDTH,),
                        activation='relu'),
                tf.keras.layers.Dense(units=32, activation='relu'),
                tf.keras.layers.Dense(units=1),
        ])
    
    def TFNet(self):
        pass