## TFNet Train Model 
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model as m
import window as w
import tensorflow as tf
#import test_model as m



#####################################
## Import the dataset


filepath = "../Data/Raw/df_week.csv"
#filepath = "../Data/one_day_testing.csv"
data = pd.read_csv(filepath)

## Lat: Latitude
## Lon: Longitude
## Time: 
## Level: hecto-Pascals
## z:  geopotential                     | Proportional to the height of a pressure level    | [m^(2) s^(−2)]            | 13 levels
## pv: potential_vorticity              | Potential vorticity                               | [K m^(2) kg^(-1) s^(-1)]  | 13 levels
## r: relative_humidity                 | Humidity relative to saturation                   | [%]                       | 13 levels
## q: specific_humidity                 | Mixing ratio of water vapor                       | [kg kg^(−1)]              | 13 levels
## t: temperature                       | Temperature                                       | [K]                       | 13 levels
## u: u_component_of_wind               | Wind in x/longitude-direction                     | [m s^(-1)]                | 13 levels
## vo: vorticity                        | Relative horizontal vorticity                     | [1 s^(-1)]                | 13 levels
## v: v_component_of_wind               | Wind in y/latitude direction                      | [m s^(-1)]                | 13 levels
## u10: 10m_u_component_of_wind         | Wind in x/longitude-direction at 10 m height      | [m s^(-1)]                | Sinlge
## v10: 10m_v_component_of_wind         | Wind in y/latitude-direction at 10 m height       | [m s^(-1)]                | Sinlge
## t2m: 2m_temperature                  | 2m_temperature                                    | [K]                       | Single
## tisr: toa_incident_solar_radiation   | Accumulated hourly incident solar radiation       | [J m^(-2)]                | Sinlge
## tcc: total_cloud_cover               | Fractional cloud cover                            | (0-1)                     | Sinlge
## tp: total_precipitation              | Hourly precipitation                              | [m]                       | Single


    ## Positive u_component_of_wind --> Wind coming from the West
    ## Negative u_component_of_wind --> Wind coming from the East
    ## Positive v_component_of_wind --> Wind coming from the South
    ## Negative v_component_of_wind --> Wind coming from the North
    ## Wind Speed = sqrt(U*U + V*V)
    ## Wind Direction Angle = arctan(V/U


#####################################


#####################################
## Clean the dataset 


## Drop Columns
    ## 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'
data = data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis = 1)
data = data.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis = 1)

## Useful Indexing:
    ## Same lat, log, level --> Unamed: 0_x
    ## Same lat, log, time --> Unamed: 0_y

## Check for null values
data = data.dropna()


#####################################


#####################################
## Raw Data Visualization


# Find lat, long, and level and compare the graph the u_component_of_wind
"""lat, long, low_level, high_level = 47.8125, 67.5, 50, 1000

low_df = data[(data["lat"] == lat) & (data["lon"] == long) & (data["level"] == low_level)] 
low_df.plot(x = "time", y = "u")
plt.ylabel("u_component_of_wind")
plt.xlabel("time")


low_df.plot(x = "time", y = "tcc")
plt.ylabel("Total Cloud Cover")
plt.xlabel("time")

low_df.plot(x = "time", y = "tp")
plt.ylabel("Total Percipation")
plt.xlabel("time")

high_df = data[(data["lat"] == lat) & (data["lon"] == long) & (data["level"] == high_level)] 
high_df.plot(x = "time", y = "u")
plt.ylabel("u_component_of_wind")
plt.xlabel("time")
plt.show()"""


#####################################


#####################################
## Preprocessing 


## Generating validation and test dataset
start_test_date = "2018-01-07 00:00:00"
end_test_date = "2018-01-07 23:00:00"

train_data = data[start_test_date < data["time"]]
test_data = data[(start_test_date >= data["time"])]

#date_time = pd.to_datetime(train_data.pop('time'), format= "%Y-%m-%d %H:%M:%S")
date_time = pd.to_datetime(train_data['time'], format= "%Y-%m-%d %H:%M:%S")
#location = 

## Raw Data Visualization
"""plot_cols = ['z', 'pv', 'r', 'q', 't', 'u', 'vo', 'v']
plot_features = train_data[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)
plt.show()"""

## Data Transposing Visualization
print(train_data.describe().transpose())
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
train_data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
train_data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
train_data['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
train_data['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# visualization of new attributes
"""plt.plot(np.array(train_data['Day sin'])[:500])
plt.plot(np.array(train_data['Day cos'])[:500])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
plt.show()"""

## Visualization of geolocation
"""plt.figure(figsize = (10,7))
sns.scatterplot(data=train_data, x='lon', y='lat')
plt.show()"""

## most important frequency features
"""fft = tf.signal.rfft(train_data['t'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(train_data['t'])
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
moving_avg = 5

## Categorize u10, v10, t2m, tisr, tcc, tp by location and time --> Normalize separately 
single_level = train_data[train_data['level'] == 1000][['lat', 'lon', 'time', 'u10', 'v10', 't2m', 'tisr', 'tcc', 'tp']]
preserved_single = single_level.head(moving_avg)
single_level = single_level.reset_index().set_index('time').groupby(['lat', 'lon']).rolling(window=moving_avg).mean()
single_level = single_level.reset_index()
#print(preserved_single)
#print(single_level.head(10))
#print('\n\n')

## Split data by level, lat, and log --> normalize each by their respective location, level, and time --> put back together
multi_level = train_data[train_data['level'] == 1000][['lat', 'lon', 'time', 'level', 'z', 'pv', 'r', 'q', 't', 'u', 'vo', 'v']]
preserved_multi = multi_level.head(moving_avg)
multi_level = multi_level.reset_index().set_index('time').groupby(['lat', 'lon', 'level']).rolling(window=moving_avg).mean()
multi_level = multi_level.reset_index()
#print(preserved_multi)
#print(multi_level.head(10))

## Merging the normalized datasets back together
train_data = pd.merge(multi_level, single_level,  how='left', left_on=['lat', 'lon', 'time'], right_on = ['lat', 'lon', 'time'])
train_data = train_data.dropna()
print(train_data.head(10))
print()

## Split data into training and validation datasets
start_val_date = "2018-01-06 00:00:00"
end_val_date = "2018-01-06 23:00:00"

train_data = train_data[train_data["time"] <= start_val_date ]
validation_data = train_data[train_data["time"] > start_val_date]

print("Train")
print(train_data)
print("-"*60 + '\n')
print("Validation")
print(validation_data)
print("-"*60 + '\n')
print("Test")
print(test_data)
print("-"*60 + '\n')

## Drop cos sin day and year from validation dataset ?

## Data Windowing
#hours = Calculate how many hours are within the dataset
hours = 7 * 24
w1 = w.WindowGenerator(input_width=hours, label_width=1, shift=24,
                     train_df= train_data, val_df= validation_data,  
                     test_df= test_data, label_columns=['t'])
#print(w1)
#days = Calculate how many days are within the dataset
days = 7 * 24
w2 = w.WindowGenerator(input_width=days, label_width=1, shift=24,
                     train_df= train_data, val_df= validation_data,  
                     test_df= test_data, label_columns=['t'])
#print(w2)

## Splitting Data Window
# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_data[:w2.total_window_size]),
                           np.array(train_data[100:100+w2.total_window_size]),
                           np.array(train_data[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

## Visualize Windowing and Splits

## Create tf.data.dataset


#####################################


#####################################
## Train Model


model = m.Model(train_data, validation_data, test_data)

## base model
#base_model = model.base()

## TFNet
tfnet_model = model.tfnet()


#####################################


#####################################
## Testing Evaluation & Visualization





#####################################


