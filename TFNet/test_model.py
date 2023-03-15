## TFNet Model
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


class Model():
    ## User picks lat, lon, time
    ## Sea Level: level = 1000
    ## Set level == 1000 --> temperature (t) & relative humidity (r) set to level 1000
    ## total cloud cover (tcc) & total prepicpation (tp) single level no need to set

    ## Predictive Attributes
    #pred_cols = ['lat', 'lon', 'time', 'level', 'z', 'pv', 'r', 'q', 't',
    #    'u', 'vo', 'v', 'u10', 'v10', 't2m', 'tisr', 'tcc',
    #    'tp']

    ## Target Attributes
    #tar_cols = ['lat', 'lon', 'time', 'level', 't', 'r', 'tcc', 'tp']
    #tar_level = 1000
    #target_cols = ['lat', 'lon', 'time', 't2m', 'tcc', 'tp']

    def __init__(self, train, validation, test) -> None:
        seed = 4
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.baseTrain = train
        self.baseValidation = validation
        self.baseTest = test

    def dataConversion(self):
        ## Positive u_component_of_wind --> Wind coming from the West
        ## Negative u_component_of_wind --> Wind coming from the East
        ## Positive v_component_of_wind --> Wind coming from the South
        ## Negative v_component_of_wind --> Wind coming from the North
        ## Wind Speed = sqrt(U*U + V*V)
        ## Wind Direction Angle = arctan(V/U)
        pass

    def base(self):
        date_time = pd.to_datetime(self.baseTrain.pop('time'), format= "%Y-%m-%d %H:%M:%S")
        #location = 

        ## Raw Data Visualization
        """plot_cols = ['z', 'pv', 'r', 'q', 't', 'u', 'vo', 'v']
        plot_features = self.train[plot_cols]
        plot_features.index = date_time
        _ = plot_features.plot(subplots=True)
        plt.show()"""

        ## Data Transposing Visualization
        print(self.baseTrain.describe().transpose())
        print()


        ## Feature Engineering
        ## Columns u, v in vector form already

        ## Converting time into seconds
        timestamp_s = date_time.map(pd.Timestamp.timestamp)

        ## Converting timestamps_s into periodicity interpertable info
        hour = 60*60
        day = 24*hour
        year = 365.2425 * day 

        #self.baseTrain['Hour sin'] = np.sin(timestamp_s * (2 * np.pi / hour))
        #self.baseTrain['Hour cos'] = np.cos(timestamp_s * (2 * np.pi / hour))
        self.baseTrain['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        self.baseTrain['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        self.baseTrain['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        self.baseTrain['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

        # visualization of new attributes
        """plt.plot(np.array(self.baseTrain['Day sin'])[:500])
        plt.plot(np.array(self.baseTrain['Day cos'])[:500])
        plt.xlabel('Time [h]')
        plt.title('Time of day signal')
        plt.show()"""

        ## Visualization of geolocation
        """plt.figure(figsize = (10,7))
        sns.scatterplot(data=self.baseTrain, x='lon', y='lat')
        plt.show()"""

        ## most important frequency features
        """fft = tf.signal.rfft(self.baseTrain['t'])
        f_per_dataset = np.arange(0, len(fft))

        n_samples_h = len(self.baseTrain['t'])
        hours_per_year = 24*365.2524
        years_per_dataset = n_samples_h/(hours_per_year)

        f_per_year = f_per_dataset/years_per_dataset
        plt.step(f_per_year, np.abs(fft))
        plt.xscale('log')
        plt.ylim(0, 4000000)
        plt.xlim([0.1, max(plt.xlim())])
        plt.xticks([1, 12, 52, 365.2524, 24*365.2524], labels=['1/Year', '1/month', '1/week', '1/day', '1/hour'])
        _ = plt.xlabel('Frequency (log scale)')
        plt.show()"""

        ## Normalize the dataset
        # remove lat and lon columns before normalization


        ## Categorize u10, v10, t2m, tisr, tcc, tp by location and time --> Normalize separately 
        ## Split data by level, lat, and log --> normalize each by their respective location, level, and time --> put back together
        ## Add u10, v10, t2m, tisr, tcc, tp back into the dataset



        base_train_ltlg = self.baseTrain[['lat', 'lon', 'level']]
        self.baseTrain = self.baseTrain.drop(['lat', 'lon', 'level'], axis = 1)
        #base_validation_ltlg = self.baseValidation.pop(['lat', 'lon'])
        #base_test_ltlg = self.baseTest.pop(['lat', 'lon'])

        # normalize datasets with moving average
        base_train_f5 = self.baseTrain.head(5)
        base_train = self.baseTrain.rolling(window=5).mean()
        print(base_train.head(10))

        # add lat and lon columns after normalization
        base_train = None 
        base_validation = None
        base_test = None

        ## Split into training and validation

        ## Data Windowing



        ## Encoder --> keep it in a single form --> throw it into time series --> done



        ## **** Add Smoothing *******


    def tfnet(self):
        pass













"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import IPython
import IPython.display
import matplotlib as mpl
import seaborn as sns


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
        
        date_time = pd.to_datetime(self.train.pop('time'), format= "%Y-%m-%d %H:%M:%S")

        ## Positive u_component_of_wind --> Wind coming from the West
        ## Negative u_component_of_wind --> Wind coming from the East
        ## Positive v_component_of_wind --> Wind coming from the South
        ## Negative v_component_of_wind --> Wind coming from the North
        ## Wind Speed = sqrt(U*U + V*V)
        ## Wind Direction Angle = arctan(V/U)




        ### FROM TENSORFLOW TUTORIAL

        CONV_WIDTH = None

        conv_window = WindowGenerator(
            input_width=CONV_WIDTH,
            label_width=1,
            shift=1,
            label_columns=['T (degC)'])

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

        history = compile_and_fit(conv_model, conv_window)

        IPython.display.clear_output()
        val_performance['Conv'] = conv_model.evaluate(conv_window.val)
        performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

        LABEL_WIDTH = 24
        INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
        wide_conv_window = WindowGenerator(
            input_width=INPUT_WIDTH,
            label_width=LABEL_WIDTH,
            shift=1,
            label_columns=['T (degC)'])

        print(wide_conv_window)

        print("Wide conv window")
        print('Input shape:', wide_conv_window.example[0].shape)
        print('Labels shape:', wide_conv_window.example[1].shape)
        print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

        wide_conv_window.plot(conv_model)
        plt.show()


















class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])"""