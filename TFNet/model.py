## TFNet Model
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class Model():
    def __init__(self, train, validation, test) -> None:
        seed = 4
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.train = train
        self.validation = validation
        self.test = test