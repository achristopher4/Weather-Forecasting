#import main as Ma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import Window as w
import BaseModel as bm
import LinearModel as lm
import CNNModel as cnn


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

column_indices = {name: i for i, name in enumerate(train_data.columns)}

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

model = bm.Model(train_data, validation_data, test_data)

single_step = model.singleStep()

## Base Model
print("\n"+ "-"*60 + "\Base Model\n")
baseline = bm.Baseline(label_index = column_indices['t'])
baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

# plotting baseline model
wide_window = w.WindowGenerator(
    input_width=24, label_width=24, shift=1,
    train_df= train_data,  val_df= validation_data,  
    test_df= test_data, label_columns=['t'])

print('\nWide Window')
print(wide_window)

print('\nTesting')
print(wide_window)
print()

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

#wide_window.plot(baseline)
#plt.show()

## Linear Model
print("\n"+ "-"*60 + "\nLinear Model\n")
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)

linearModel = lm.LinearModel()

history = linearModel.compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

wide_window.plot(linear)
plt.show()

plt.bar(x = range(len(train_data.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_data.columns)))
_ = axis.set_xticklabels(train_data.columns, rotation=90)
plt.show()

## CNN Model
#cnn_model = cnn.CNNModel()

## TFNet Model
#tfnet_model = model.TFNet()


#####################################


#####################################
## Testing Evaluation & Visualization





#####################################



