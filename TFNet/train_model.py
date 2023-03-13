## TFNet Train Model 
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#####################################
## Import the dataset

filepath = "../Data/df_week.csv"
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

#####################################


#####################################
## Clean the dataset 

## Drop Columns
    ## 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'
data = data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis = 1)
#data = data.drop(['Unamed: 0_x', 'Unamed: 0_y'], axis = 1)

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

high_df = data[(data["lat"] == lat) & (data["lon"] == long) & (data["level"] == high_level)] 
high_df.plot(x = "time", y = "u")
plt.ylabel("u_component_of_wind")
plt.xlabel("time")
#plt.show()

#####################################


#####################################
## Preprocessing 

#####################################


#####################################
## Run through Model

## Predictive Attributes
pred_cols = ['lat', 'lon', 'time', 'level', 'z', 'pv', 'r', 'q', 't',
       'u', 'vo', 'v', 'u10', 'v10', 't2m', 'tisr', 'tcc',
       'tp']

## Target Attributes
tar_cols = ['lat', 'lon', 'time', 'level', 't', 'r', 'tcc', 'tp']
tar_level = 1000

## User picks lat, lon, time
## Sea Level: level = 1000
## Set level == 1000 --> temperature (t) & relative humidity (r) set to level 1000
## total cloud cover (tcc) & total prepicpation (tp) single level no need to set



#####################################


#####################################
## Result Visualization