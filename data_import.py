# This is a sample Python script.
import xarray
import os
import netCDF4
import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


main_dir = os.path.abspath('H:/Users/obsid/Documents/Weather Data/')
directories = os.listdir(main_dir)
for sub_dir in directories:
    print(sub_dir)
    if sub_dir == "geopotential_500" or sub_dir == "temperature_850":
        dir = os.path.abspath('H:/Users/obsid/Documents/Weather Data/' + sub_dir + '/' + sub_dir + 'hPa_2018_5.625deg.nc')
    elif sub_dir != "constants":
        dir = os.path.abspath('H:/Users/obsid/Documents/Weather Data/' + sub_dir + '/' + sub_dir + '_2018_5.625deg.nc')
    ds = xarray.open_dataset(dir, engine="netcdf4")
    #df = ds.to_dataframe()
    cords = list(ds.coords)
    if sub_dir == "10m_u_component_of_wind":
        df_3 = ds.to_dataframe()
    elif sub_dir == "geopotential":
        df_4 = ds.to_dataframe()
    elif len(cords) == 3:
        df_temp = ds.to_dataframe()
        df_3 = df_3.merge(df_temp, on=["lat", "lon", "time"])
    elif len(cords) == 3:
        df_temp = ds.to_dataframe()
        df_4 = df_4.merge(df_temp, on=["lat", "lon", "time", "level"])

df_3 = df_3.reset_index()
df_4 = df_4.reset_index()

df_3.to_csv("df_3.csv")
df_4.to_csv("df_4.csv")


#ds = xarray.open_dataset(dir, engine="netcdf4")

#print(ds)

"geopotential/geopotential_2018_5.625deg.nc"