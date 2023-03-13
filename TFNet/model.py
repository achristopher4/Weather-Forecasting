## TFNet Model
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class CNN_Model():
    def __init__(self, data, cols, target) -> None:
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.data = data
        self.pred_cols = cols
        self.target_level = 1000
        self.target_cols = target
    
    def initalize_model(self):
        pass