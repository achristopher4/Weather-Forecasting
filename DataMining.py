## Weather Data Extraction
## Date: 1/15/2023

## Website: http://www.climate.psu.edu/data/current/dailysum.php

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import pandas as pd

## Source Point
source = "http://www.climate.psu.edu/data/current/dailysum.php?id=KPNE"

## Web Scraping
driver = webdriver.Firefox()
driver.get(source) 

dataframe = pd.DataFrame()
last_yeat = None

for year in range(1, 24):
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "linkstable")))
    time.sleep(0.1)

    table = driver.find_elements(by = By.TAG_NAME, value = "table")
    pandas_readable = table[0].get_attribute('outerHTML')
    yearData = pd.read_html(pandas_readable)[0]
    dataframe = pd.concat([dataframe, yearData])
    time.sleep(0.01)

    select_year = driver.find_element(by = By.TAG_NAME, value = "center").find_elements(by = By.TAG_NAME, value = "select")[-1]
    select_year.click()
    time.sleep(0.1)
    selected_option = select_year.find_elements(by = By.TAG_NAME, value = "option")
    last_year = selected_option[year].text
    selected_option[year].click()
    time.sleep(0.1)

    submit = driver.find_element(by = By.TAG_NAME, value = "center").find_element(by = By.TAG_NAME, value = "input")
    submit.click()
    time.sleep(0.01)
driver.close()

## Data Cleaning
#  Drop all rows that have subheaders
dataframe = dataframe[dataframe['Date'] != 'Date']

## Export DataFrame
filename = "PSU_Climate_Weather_" + last_year + "_2023.csv"
path = "./Data/" + filename
dataframe.to_csv(path)

"""
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "linkstable")))
time.sleep(0.1)

#table = driver.find_element(by = By.CLASS_NAME, value = "linkstable")
table = driver.find_elements(by = By.TAG_NAME, value = "table")
pandas_readable = table[0].get_attribute('outerHTML')
dataframe = pd.read_html(pandas_readable)[0]
"""

