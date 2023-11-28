from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr

import time
import pandas as pd
#allows scrolling to happen in selenium to gather all data. Huy Tran - Thompson

def scrolling_within_element(driver, element_css_selector):
    scrollable_element = driver.find_element(By.CLASS_NAME, element_css_selector)
    last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_element)

    while True:
        driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight);", scrollable_element)
        time.sleep(3) 
        new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_element)
        if new_height == last_height:
            break
        last_height = new_height



#Create a selenium driver for operation. Thompson. 
def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    driver = webdriver.Chrome(options=options)
    return driver

#Scraping world data using beautiful Soup and Selenium. Thompson - Huy Tran
def scraping_world_bank_data(driver, url):
    driver.get(url)
    driver.implicitly_wait(10)
    scrolling_within_element(driver, 'dxgvCSD')
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', attrs={'class': "dxgvControl_GridDefaultTheme dxgv"})
    if table:
        df = pd.read_html(str(table))[0]
        df.drop(df.columns[[1, 2, 12]], axis=1, inplace=True)
        df = df.iloc[19:234]
        return df
    else:
        print("Table not found")
        return None