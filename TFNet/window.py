## Data Windowing
## Date: 16 March 2023
## Contributors: Alexander Christopher


import pandas as pd

"""
    Make a set of predictions based on a window of consecutive samples from the data.
        The width (number of time steps) of the input and label windows.
        The time offset between them.
        Which features are used as inputs, labels, or both.
    Can be used for:
        Single-output, and multi-output predictions.
        Single-time-step and multi-time-step predictions.
"""

class DataWindow():
    ## WindowGenerator:
        ## Handle the indexes and offsets as shown in the diagrams above.
        ## Split windows of features into (features, labels) pairs.
        ## Plot the content of the resulting windows.
        ## Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.
    def __init__(self):
        ## Includes all the necessary logic for the input and label indices.
        pass

    #def __repr__(self):
    #    pass