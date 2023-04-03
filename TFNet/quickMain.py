#import main as Ma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Model as m
import window as w
import tensorflow as tf

import BaseModel as bm

SEED = 44
tf.random.set_seed(SEED)
np.random.seed(SEED)

#####################################
## Import Preprocessed Data

preprocess_export_path = "../Data/Preprocessing/"
train_data = pd.read_csv(preprocess_export_path + 'train.csv').drop(['Unnamed: 0', 'index_x', 'index_y'], axis = 1)
validation_data = pd.read_csv(preprocess_export_path + 'validation.csv')
test_data = pd.read_csv(preprocess_export_path + 'test.csv')

#####################################


#####################################
trainDateTime = train_data.pop('time')
valDateTime = validation_data.pop('time')
testDateTime = test_data.pop('time')

"""
Idea:

    Segement the train_data into separate dataframes by lat and long
    Use tf.stack to stack the dataframe ontop of each other

"""

## Data Windowing

## Splitting Data Window
#hours = Calculate how many hours are within the dataset
hours = 7 * 24

""" Testing for multi lat, longs, levels """
#latLong = train_data[['lat', 'long']].nunique()
#levels = 13

w1 = w.WindowGenerator(input_width=hours, label_width=1, shift=24,
                     train_df= train_data, val_df= validation_data,  
                     test_df= test_data, label_columns=['t'])

#days = Calculate how many days are within the dataset
days = 7 * 24
w2 = w.WindowGenerator(input_width=days, label_width=1, shift=24,
                     train_df= train_data, val_df= validation_data,  
                     test_df= test_data, label_columns=['t'])


## Visualize Windowing and Splits
example_window = tf.stack([np.array(train_data[:w2.total_window_size], dtype=np.float),
                           np.array(train_data[100:100+w2.total_window_size], dtype=np.float),
                           np.array(train_data[200:200+w2.total_window_size], dtype=np.float)])

example_inputs, example_labels = w2.split_window(example_window)

print('\nAll shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}\n')

## Create tf.data.dataset
#tf_dataset = w.WindowGenerator.make_dataset(train_data)

## Window Plot Visualization
w2.example = example_inputs, example_labels
#w2.plot()
#plt.show()

# Each element is an (inputs, label) pair.
w2_viz = w2.train.element_spec
print(f"\nEach element is an (inputs, label) pair.\n{w2_viz}\n")

for example_inputs, example_labels in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

#####################################


#####################################
## Train Model


print(f"\nSingle Step Model")
single_step_window = w.WindowGenerator(input_width= 1, label_width= 1, 
                                        shift= 1, train_df= train_data, 
                                        val_df= validation_data,  
                                        test_df= test_data,
                                        label_columns= ['t'])
print(f"{single_step_window}\n")

for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
print()

model = m.Model(train_data, validation_data, test_data)

single_step = model.singleStep()

## base model
#base_model = model.base()

## TFNet
#tfnet_model = model.TFNet()


#####################################


#####################################
## Testing Evaluation & Visualization





#####################################



