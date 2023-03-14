## TFNet Train Model 
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import model as m
import test_model as m



#####################################
## Import the dataset


filepath = "../Data/df_week.csv"
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
## Visualization


# Find lat, long, and level and compare the graph the u_component_of_wind
lat, long, low_level, high_level = 47.8125, 67.5, 50, 1000

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
#plt.show()


#####################################


#####################################
## Preprocessing 


## Generating validation set
start_val_date = "2018-01-07 00:00:00"
end_val_date = "2018-01-07 23:00:00"

train = data[start_val_date > data["time"]]
validation = data[(start_val_date <= data["time"]) & (data["time"] <= end_val_date)]


#####################################


#####################################
## Train Model


## User picks lat, lon, time
## Sea Level: level = 1000
## Set level == 1000 --> temperature (t) & relative humidity (r) set to level 1000
## total cloud cover (tcc) & total prepicpation (tp) single level no need to set

## Predictive Attributes
pred_cols = ['lat', 'lon', 'time', 'level', 'z', 'pv', 'r', 'q', 't',
       'u', 'vo', 'v', 'u10', 'v10', 't2m', 'tisr', 'tcc',
       'tp']

## Target Attributes
#tar_cols = ['lat', 'lon', 'time', 'level', 't', 'r', 'tcc', 'tp']
#tar_level = 1000
target_cols = ['lat', 'lon', 'time', 't2m', 'tcc', 'tp']

## base model
m.Model(train, validation, pred_cols, target_cols)


#####################################


#####################################
## Testing Evaluation & Visualization





#####################################


