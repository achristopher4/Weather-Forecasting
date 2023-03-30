## Final Iteration of TFNet Model
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import IPython
import IPython.display
import matplotlib as mpl
import seaborn as sns

SEED = 44

class Model():
    def __init__(self, train, validation, test) -> None:
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        self.baseTrain = train
        self.baseValidation = validation
        self.baseTest = test
    
    def baseCNN(self):
        pass
    
    def TFNet(self):
        pass