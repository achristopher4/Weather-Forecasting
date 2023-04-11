#import main as Ma

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython
import IPython.display

import Window as w
import BaseModel as bm
import LinearModel as lm
import DenseModel as dm
import MSDenseModel as msd
import CNNModel as cnn


SEED = 44
tf.random.set_seed(SEED)
np.random.seed(SEED)

#####################################
## Import Preprocessed Data

preprocess_export_path = "../Data/Preprocessing/"
train_data = pd.read_csv(preprocess_export_path + 'train.csv').drop(['Unnamed: 0', 'index_x', 'index_y'], axis = 1)
validation_data = pd.read_csv(preprocess_export_path + 'validation.csv').drop(['Unnamed: 0', 'index_x', 'index_y'], axis = 1)
test_data = pd.read_csv(preprocess_export_path + 'test.csv').drop(['Unnamed: 0'], axis = 1)

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


## Base Model
print("\n"+ "-"*60 + "\nBase Model\n")
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

"""

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


## Dense Model
print("\n"+ "-"*60 + "\nDense Model\n")
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

denseModel = dm.DenseModel()

history = denseModel.compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

#wide_window.plot(dense)
#plt.show()


## Multi-Step Dense Model
print("\n"+ "-"*60 + "\nMulti-Step Dense Model\n")

msDense_model = msd.MSDenseModel() 

CONV_WIDTH = 3
conv_window = w.WindowGenerator(
    input_width=CONV_WIDTH, label_width=1,
    shift=1, label_columns=['t'], 
    train_df= train_data, val_df= validation_data,  
    test_df= test_data,
    )

print(conv_window)

conv_window.plot()
plt.title("Given 3 hours of inputs, predict 1 hour into the future.")

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)
print()

history = msDense_model.compile_and_fit(multi_step_dense, conv_window)

IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

print()
conv_window.plot(multi_step_dense)
plt.show()
print()

print('\nInput shape:', wide_window.example[0].shape)
try:
    print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
    print(f'\n{type(e).__name__}:{e}')
"""



## CNN Model
print("\n"+ "-"*60 + "\nCNN Model\n")
#CONV_WIDTH = 3
CONV_WIDTH = 24

conv_window = w.WindowGenerator(
    input_width=CONV_WIDTH, label_width=1,
    shift=1, label_columns=['t'], 
    train_df= train_data, val_df= validation_data,  
    test_df= test_data
    )


conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

cnn_model = cnn.CNNModel()

history = cnn_model.compile_and_fit(conv_model, conv_window)

IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)

#LABEL_WIDTH = 24
LABEL_WIDTH = 17
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = w.WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1, label_columns=['t'], 
    train_df= train_data, val_df= validation_data,  
    test_df= test_data)

print(wide_conv_window)

print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

wide_conv_window.plot(conv_model)
plt.show()



## TFNet Model
print("\n"+ "-"*60 + "\nTFNet Model\n")
#tfnet_model = model.TFNet()


#####################################


#####################################
## Testing Evaluation & Visualization





#####################################



