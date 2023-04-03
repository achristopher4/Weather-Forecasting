## Data Windowing
## Date: 16 March 2023
## Contributors: Alexander Christopher


import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt

SEED = 44
tf.random.set_seed(SEED)

"""
Overview:
    Make a set of predictions based on a window of consecutive samples from the data.
        The width (number of time steps) of the input and label windows.
        The time offset between them.
        Which features are used as inputs, labels, or both.
    Can be used for:
        Single-output, and multi-output predictions.
        Single-time-step and multi-time-step predictions.
"""

class WindowGenerator():
    ## WindowGenerator:
        ## Handle the indexes and offsets as shown in the diagrams above.
        ## Split windows of features into (features, labels) pairs.
        ## Plot the content of the resulting windows.
        ## Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.
    def __init__(self, input_width, label_width, shift, train_df, 
                    val_df, test_df, label_columns=None):
        ## Includes all the necessary logic for the input and label indices.
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        
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

        #
        self.examlple = None
        #

        ## Store raw data
        self.train = train_df
        self.val = val_df
        self.test = test_df

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot(self, model=None, plot_col='t', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)
        return ds
    
    @property
    def train(self):
        return self._train
    
    @train.setter
    def train(self, train):
        self._train = self.make_dataset(train)

    @property
    def val(self):
        return self._val
    
    @val.setter
    def val(self, val):
        self._val = self.make_dataset(val)

    @property
    def test(self):
        return self._test
    
    @test.setter
    def test(self, test):
        self._test = self.make_dataset(test)

    ############ TESTING ###############

    @property
    def example(self):
        return self._example

    @example.setter
    def example(self, inputs = None, labels = None):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        self._example = result
