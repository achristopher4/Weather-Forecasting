## Data Windowing
## Date: 16 March 2023
## Contributors: Alexander Christopher


import pandas as pd
import numpy as np

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
        
        ## Store raw data
        self.train = train_df
        self.val = val_df
        self.test = test_df

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
            f'Label column name(s): {self.label_columns}'])