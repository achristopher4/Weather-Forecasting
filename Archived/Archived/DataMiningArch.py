## Weather Data Extraction
## Date: 1/15/2023

## Website: https://www.wunderground.com/history/monthly/us/pa/state-college


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import pandas as pd

## Source Point
source = "https://www.wunderground.com/history/monthly/us/pa/state-college/KUNV/date/2022-1" 

driver = webdriver.Firefox()
driver.get(source) 


WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "cdk-overlay-pane")))
time.sleep(0.1)
priv_pol_pop_up = driver.find_element(by = By.CLASS_NAME, value = "cdk-overlay-pane")
priv_pol_pop_up.find_element(by = By.CLASS_NAME, value = "close").click()
time.sleep(0.01)

mainColumns = ["Month", "Day", "Year", "Temperature (°F) ", "Dew Point (°F) ", "Humidity (%) ", "Wind Speed (mph) ", 
                "Pressure (in) ", "Precipitation (in) "]
subColumns = ["Max", "Avg", "Min"]

main_table = driver.find_element(by = By.CLASS_NAME, value = "days.ng-star-inserted")
sub_tables_elements = main_table.find_elements(by = By.TAG_NAME, value = "table")

df = pd.DataFrame()

for st in range(len(sub_tables_elements)):
    pandas_readable = sub_tables_elements[st].get_attribute('outerHTML')
    sub_table = pd.read_html(pandas_readable)[0]
    if st > 0:
        newColumnNames = {}
        baseName = mainColumns[st + 2]
        currentNames = sub_table.columns
        newNames = list(sub_table.iloc[0])
        sub_table = sub_table.drop(0)
        for c in range(len(currentNames)):
            newColumnNames[currentNames[c]] = baseName + newNames[c]
        sub_table = sub_table.rename(columns = newColumnNames)
    else:
        sub_table = pd.DataFrame(sub_table)
        newNames = list(sub_table.iloc[0])
        sub_table = sub_table.drop(0)
        sub_table = sub_table.rename(columns= {0:newNames[0]})

    df = pd.concat([df, sub_table], axis = 1)
print(df)



