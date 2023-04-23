## Date: 4/23/2023
## Time Series Weather Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython
import IPython.display

import Window as w
import ModelConstructor as mc


SEED = 44
tf.random.set_seed(SEED)
np.random.seed(SEED)

##########################################################################
## Data Selection and Importation 

preprocess_export_path = "../../Data/Single_Point_Preprocessing/"


train_data = pd.read_csv(preprocess_export_path + 'train.csv').drop(['Unnamed: 0'], axis = 1)
validation_data = pd.read_csv(preprocess_export_path + 'validation.csv').drop(['Unnamed: 0'], axis = 1)
test_data = pd.read_csv(preprocess_export_path + 'test.csv').drop(['Unnamed: 0'], axis = 1)

##########################################################################



##########################################################################
## Final Preprocessing

"""
Experiment: Drop 
    train_data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    train_data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    train_data['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    train_data['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

"""

trainDateTime = train_data.pop('time')
valDateTime = validation_data.pop('time')
testDateTime = test_data.pop('time')

column_indices = {name: i for i, name in enumerate(train_data.columns)}

"""
Idea:

    Segement the train_data into separate dataframes by lat and long
    Use tf.stack to stack the dataframe ontop of each other
"""

## Drop all rows in validation & testing not at ground level
validation_data = validation_data[validation_data['level'] == 1000]
test_data = validation_data[validation_data['level'] == 1000]

##########################################################################



##########################################################################
## Window Prep

CONV_WIDTH = 24
LABEL_WIDTH = 17
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)

## Winodw Single Variable
single_step_window = w.WindowGenerator(input_width= 1, label_width= 1, 
                                        shift= 1, train_df= train_data, 
                                        val_df= validation_data,  
                                        test_df= test_data,
                                        label_columns= ['t'])

## Wide Window Single Variable ('t')
wide_window = w.WindowGenerator(
    input_width=24, label_width=24, shift=1,
    train_df= train_data,  val_df= validation_data,  
    test_df= test_data, label_columns=['t'])

## Wide Window Multi Variable ('t', 'r', 'u', 'v', 'tp', 'tcc', 'tisr')
mv_wide_window = w.WindowGenerator(
    input_width=24, label_width=24, shift=1,
    train_df= train_data,  val_df= validation_data,  
    test_df= test_data, label_columns=['t', 'r', 'u', 'v', 'tp', 'tcc', 'tisr'])

## Convoluation Window
conv_window = w.WindowGenerator(
    input_width=CONV_WIDTH, label_width=1,
    shift=1, label_columns=['t'], 
    train_df= train_data, val_df= validation_data,  
    test_df= test_data,
    )

## Wide Convoluation Window
wide_conv_window = w.WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1, label_columns=['t'], 
    train_df= train_data, val_df= validation_data,  
    test_df= test_data)

##########################################################################



##########################################################################
## Other Model Prep

val_performance = {}
performance = {}

##########################################################################



##########################################################################
## Train Model
"""

###########           Base Model            #############
print("\n"+ "-"*60 + "\nBase Model\n")
baseline = mc.ModelConstructor(label_index = column_indices['t'])
baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}

val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

# plotting baseline model
print('\nWide Window')
print(wide_window)

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

wide_window.plot(baseline)
plt.title("Base Model")
plt.show()
#########################################################

###########     Single Step Linear Model    #############
print("\n"+ "-"*60 + "\nLinear Model\n")
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)

linearModel = mc.ModelConstructor()

history = linearModel.compile_and_fit(linear, single_step_window, model_type_name = "Single Step Linear Model")

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

wide_window.plot(linear)
plt.title("Linear Model")
plt.show()

plt.bar(x = range(len(train_data.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_data.columns)))
_ = axis.set_xticklabels(train_data.columns, rotation=90)
plt.title("Linear Model Weights")
plt.show()
#########################################################

###########     Single Step Dense Model     #############
print("\n"+ "-"*60 + "\nDense Model\n")
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

denseModel = mc.ModelConstructor()

history = denseModel.compile_and_fit(dense, single_step_window, model_type_name = "Single Step Dense")

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

wide_window.plot(dense)
plt.title("Dense Model")
plt.show()
#########################################################

###########     Multi Step Dense Model      #############
print("\n"+ "-"*60 + "\nMulti-Step Dense Model\n")

msDense_model = mc.ModelConstructor() 

print(conv_window)

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

history = msDense_model.compile_and_fit(multi_step_dense, conv_window, model_type_name = "Multi-Step Dense")

IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

print()
conv_window.plot(multi_step_dense)
plt.title("Multi-Step Dense Model")
plt.show()
print()

print('\nInput shape:', wide_window.example[0].shape)
try:
    print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
    print(f'\n{type(e).__name__}:{e}')
"""
#########################################################

###########             CNN Model           #############
print("\n"+ "-"*60 + "\nCNN Model\n")

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    #tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

cnn_model = mc.ModelConstructor()

history = cnn_model.compile_and_fit(conv_model, conv_window, patience = 10, model_type_name = "CNN")

IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)

print(wide_conv_window)

print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

wide_conv_window.plot(conv_model)
plt.title("CNN Model")
plt.show()

#########################################################

###########             LSTM Model           #############

#########################################################

##########################################################################


