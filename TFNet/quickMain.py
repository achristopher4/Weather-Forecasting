#import main as Ma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model as m
import window as w
import tensorflow as tf

#####################################
## Import Preprocessed Data

preprocess_export_path = "../Data/Preprocessing/"
train_data = pd.read_csv(preprocess_export_path + 'train.csv').drop(['Unnamed: 0', 'index_x', 'index_y'], axis = 1)
validation_data = pd.read_csv(preprocess_export_path + 'validation.csv')
test_data = pd.read_csv(preprocess_export_path + 'test.csv')

#####################################


#####################################
dateTime = train_data.pop('time')

## Data Windowing
#hours = Calculate how many hours are within the dataset
hours = 7 * 24
w1 = w.WindowGenerator(input_width=hours, label_width=1, shift=24,
                     train_df= train_data, val_df= validation_data,  
                     test_df= test_data, label_columns=['t'])
#print(w1)
#days = Calculate how many days are within the dataset
days = 7 * 24
w2 = w.WindowGenerator(input_width=days, label_width=1, shift=24,
                     train_df= train_data, val_df= validation_data,  
                     test_df= test_data, label_columns=['t'])
#print(w2)

## Splitting Data Window
# Stack three slices, the length of the total window.

print("\nSplitting the Data Window\n")
"""print(w2.total_window_size)
print()
print(np.array(train_data[:w2.total_window_size], dtype=np.float))
print(np.array(train_data[100:100+w2.total_window_size], dtype=np.float))
print(np.array(train_data[200:200+w2.total_window_size], dtype=np.float))
print()
print(train_data.columns)
print()
print(w2)
print()"""

# tf.stack([ np.array(pd.DataFrame()), np.array(pd.DataFrame()), np.array(pd.DataFrame()) ])

#"""
example_window = tf.stack([np.array(train_data[:w2.total_window_size], dtype=np.float),
                           np.array(train_data[100:100+w2.total_window_size], dtype=np.float),
                           np.array(train_data[200:200+w2.total_window_size], dtype=np.float)])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
#"""

## Visualize Windowing and Splits

## Create tf.data.dataset


#####################################


#####################################
## Train Model


model = m.Model(train_data, validation_data, test_data)

## base model
#base_model = model.base()

## TFNet
tfnet_model = model.TFNet()


#####################################


#####################################
## Testing Evaluation & Visualization





#####################################



