import xarray
import os
import netCDF4
import pandas as pd


main_dir = os.path.abspath('H:/Users/obsid/Documents/Weather Data/')
directories = os.listdir(main_dir)
df_4 = None
df_3 = None
# Excludes specific fields. The three listed below are excluded by default as they are constant values
# or subsets at a specific pressure level.
dir_exclude = ["constants", "geopotential_500", "temperature_850"]
# Gives a date range to subset the data on. Each must be given in the syntax of year-month-day.
# Leaving the list blank will skip the sub-setting phase.
dates = ["2018-01-01", "2018-01-08"]
# Loads the data into two dataframes depending on index count.
for sub_dir in directories:
    if sub_dir not in dir_exclude:
        dir = os.path.abspath('H:/Users/obsid/Documents/Weather Data/' + sub_dir + '/' + sub_dir + '_2018_5.625deg.nc')
        ds = xarray.open_dataset(dir, engine="netcdf4")
        cords = list(ds.coords)
        if len(cords) == 3 and df_3 is None:
            print("Index: 3", sub_dir)
            df_3 = ds.to_dataframe()
        elif len(cords) == 4 and df_4 is None:
            print("Index: 4", sub_dir)
            df_4 = ds.to_dataframe()
        elif len(cords) == 3:
            print("Index: 3", sub_dir)
            df_temp = ds.to_dataframe()
            df_3 = df_3.merge(df_temp, on=["lat", "lon", "time"])
        elif len(cords) == 4:
            print("Index: 4", sub_dir)
            df_temp = ds.to_dataframe()
            df_4 = df_4.merge(df_temp, on=["lat", "lon", "time", "level"])

# Merges the two dataframes on the three shared indices
print("Merging")
combined_df = df_3.merge(df_4, on=["lat", "lon", "time"])

# Flattens the index, converts time to datatime syntax, and subsets on time.
print("Formatting Columns and Sub-setting")
combined_df = combined_df.reset_index()
df = pd.DataFrame()
if len(dates) != 0:
    chunk_list = []
    chunk_size = 10 ** 6
    i = 0
    for chunk in pd.read_csv('df_subset.csv', chunksize=chunk_size):
        chunk["time"] = pd.to_datetime(chunk["time"])
        df = pd.concat([combined_df, chunk[(chunk['time'] >= dates[0]) & (chunk['time'] < dates[1])]])
        i = i+1

# Saves the dataframe to a csv file.
print("Saving to CSV")
df.to_csv("combined_df.csv")


