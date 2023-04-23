## Date: 04/23/2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf



SEED = 44
np.random.seed(SEED)


filepath = "../Data/Raw/df_point.csv"
#filepath = "../Data/one_day_testing.csv"
data = pd.read_csv(filepath)

data = data.drop(['Unnamed: 0'], axis = 1)

data = data.dropna()

## 2018-01-01 00:00:00
## 2018-12-31 23:00:00

start_test_date = "2018-12-24 00:00:00"

train_data = data[data["time"] < start_test_date ]
test_data = data[data["time"] >= start_test_date ]

start_val_date = "2018-12-01 00:00:00"

validation_data = train_data[train_data["time"] >= start_val_date ]
train_data = train_data[train_data["time"] < start_val_date ]

preprocess_export_path = "../Data/Single_Point_Preprocessing/"
train_data.to_csv(preprocess_export_path + 'train.csv')
validation_data.to_csv(preprocess_export_path + 'validation.csv')
test_data.to_csv(preprocess_export_path + 'test.csv')
