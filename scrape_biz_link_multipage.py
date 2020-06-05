import os, random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re

def setOptions():
    options = webdriver.ChromeOptions();
    options.add_argument('--disable-infobars');
    options.add_argument('--disable-dev-shm-usage');
    options.add_argument('--disable-extensions');
    options.add_argument('--headless');
    options.add_argument('--disable-gpu');
    options.add_argument('--no-sandbox');
    options.add_argument('--no-proxy-server')
    options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"]);
    return options


def startDriver():
    options = setOptions()
    driver = webdriver.Chrome(options=options);
    wait = WebDriverWait(driver, 30);
    return driver, wait

def quitDriver(driver):
    driver.close();
    driver.quit();


def startThread(descr, city):
    listOverall = []
    driver, wait = startDriver()

    url = 'https://www.yelp.com/search?find_desc=' + descr + '&find_loc='+city+'&sortby=review_count'
    print(url)

    driver.get(url)

    pageLoaded = wait.until(EC.visibility_of_element_located((By.ID,"wrap")));
    soup = BeautifulSoup(driver.page_source, 'lxml')
    currentPage = []
    page = 0

    while(True):
        print('Searching: ',city,' on page: ',page)
        for link in soup.findAll('a', class_="lemon--a__373c0__IEZFH link__373c0__1G70M link-color--inherit__373c0__3dzpk link-size--inherit__373c0__1VFlE")[0:]:
            biz_url = link.get('href')
            if biz_url[0:4] == '/biz':
                currentItem = [biz_url]
                currentPage.append(currentItem)

        try:
            nextURL = soup.find("a", class_="lemon--a__373c0__IEZFH link__373c0__1G70M next-link navigation-button__373c0__23BAT link-color--inherit__373c0__3dzpk link-size--inherit__373c0__1VFlE")["href"]
            nextURL = "https://www.yelp.com" + nextURL


            driver.get(nextURL)
            page = page + 1
            pageLoaded = wait.until(EC.visibility_of_element_located((By.ID,"wrap")));
            soup = BeautifulSoup(driver.page_source, 'lxml')
        except:
            listOverall=listOverall+currentPage
            break
    driver.quit()
    return listOverall




if __name__ == '__main__':
    import numpy as np
    import time, glob

    descrp = ['pet hotel', 'pet boarding', 'pet kennel', 'pet sitting', 'pet services']
    # zipcodes_NYC = np.loadtxt('zip_code.txt', delimiter=',')
    # zipcodes_NYC = np.loadtxt('zip_code_SF.txt', delimiter=',')
    zipcodes_NYC = list(map(int, zipcodes_NYC))

    for des in descrp:
        try:
            _des = des.replace(' ', '_')
        finally:
            for i, city in enumerate(zipcodes_NYC):
                city = str(city)

                if len(glob.glob('scraped_data/'+city+'_'+_des+'_v*.csv')) == 0:
                    listOverall = startThread(des, city)
                    output = pd.DataFrame.from_records(listOverall, columns=['url'])
                    now = time.strftime("%Y%m%d-%H%M%S")
                    output_csv = 'scraped_data/'+city+'_' + _des + '_v'+now+'.csv'
                    output.to_csv(output_csv)