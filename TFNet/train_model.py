## TFNet Train Model 
## Date: 28 Feb 2023
## Contributors: Alexander Christopher, Werner Hager

import pandas as pd
import numpy as np


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



#####################################


#####################################
## Preprocessing 

#####################################


#####################################
## Run through Model

#####################################


#####################################
## Result Visualization