
import os
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import glob
import json
import time


def setOptions():
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--no-proxy-server')
    options.add_experimental_option(
        "excludeSwitches", ["ignore-certificate-errors"])
    return options


def getNextPage(driver, wait):
    nextURL = driver.find_element_by_xpath(
        "//a[@class='u-decoration-none next pagination-links_anchor']").get_attribute('href')
    soup = finishLoadGrabSource(nextURL, driver, wait)
    return soup


def finishLoadGrabSource(url, driver, wait):

    loadingCondition = EC.invisibility_of_element_located(
        (By.CLASS_NAME, 'throbber'))
    try:
        driver.get(url)
        pageLoaded = wait.until(loadingCondition)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        return soup
    except:
        print('Page failed to load, or browser time out')
        pass


def getBusinessDetails(s):
    return [s.at['url']]


def startDriver():
    options = setOptions()
    driver = webdriver.Chrome(options=options)

    wait = WebDriverWait(driver, 30)
    return driver, wait


def quitDriver(driver):
    driver.close()
    driver.quit()


def startThread(desc, city):

#     filepath = glob.glob('scraped_data/' + city + '_businesses_v*.csv')
    filepath = glob.glob('scraped_data/' + city + '_' + desc + '_v*.csv')

    if(len(filepath) == 1):
        path = filepath[0]
        df = pd.read_csv(path)
    elif (len(filepath) > 1):
        print('WARNING: more than 1 city file for city')
        print(filepath)
        import pdb; pdb.set_trace()
        return False

    driver, wait = startDriver()
    listOverall = []

    totalPages = df.index.size
    for i in df.index:
        business = i + 1
        print('Scraping Reviews for: ', city, ' Type: ', desc,' Business: ', business)

        url = 'https://www.yelp.com' + df.at[i, 'url']
        bizList = getBusinessDetails(df.loc[i])

        if((i > 0) & (i % 20 == 0)):
            quitDriver(driver)
            driver, wait = startDriver()

        soup = finishLoadGrabSource(url, driver, wait)

        if soup is None:
            quitDriver(driver)
            time.sleep(30)
            driver, wait = startDriver()
            soup = finishLoadGrabSource(url, driver, wait)
            if soup is None:
                currentBizList = []
                bizList = [bizList[0], "NA", "NA", "NA"]
                reviewList = [ "NA", "NA", "NA", "NA"]
                currentBizList.append(reviewList + bizList)
                listOverall = listOverall + currentBizList
                continue

        for div in soup.find_all("div", class_="hidden"):
            try:
                biz_name = div.find("meta", itemprop="name")
                biz_name = biz_name['content']
            except:
                biz_name = "NA"
            try:
                biz_phone = div.find("span", itemprop="telephone").get_text()
                biz_phone = biz_phone.strip()
            except:
                biz_phone = "NA"
            try:
                biz_rating = div.find("meta", itemprop="ratingValue")
                biz_rating = biz_rating['content']
            except:
                biz_rating = "NA"

        currentBizList = []
        bizList = [bizList[0], biz_rating, biz_name, biz_phone]

        table = soup.find('tbody', class_='lemon--tbody__373c0__2T6Pl')

        try:
            opn = []
            for row in table.find_all("tr"):
                cell = row.find("td")
                hr = cell.get_text()
                opn.append(hr)

            mon = opn[0]
            tues = opn[1]
            wed = opn[2]
            thurs = opn[3]
            fri = opn[4]
            sat = opn[5]
            sun = opn[6]

        except:
            mon = "NA"
            tues = "NA"
            wed = "NA"
            thurs = "NA"
            fri = "NA"
            sat = "NA"
            sun = "NA"
        bizHrs = [mon, tues, wed, thurs, fri, sat, sun]
        bizList = bizList + bizHrs

        while(True):

            for rev in soup.find_all("div", itemprop="review"):
                try:
                    review_name = rev.find("meta", itemprop="author")
                    review_name = review_name['content']
                except:
                    review_name = "NA"
                try:
                    review_rating = rev.find("meta",  itemprop="ratingValue")
                    review_rating = review_rating['content']
                except:
                    review_rating = "NA"
                try:
                    review_date = rev.find("meta",  itemprop="datePublished")
                    review_date = review_date['content']
                except:
                    review_date = "NA"
                try:
                    review_text = rev.find(
                        'p', attrs={'itemprop': 'description'}).get_text()
                except:

                    review_text = ""
                    continue
                reviewList = [review_name, review_rating,
                              review_date, review_text]
                currentBizList.append(reviewList + bizList)

            try:
                soup = getNextPage(driver, wait)
                time.sleep(3)
            except:

                listOverall = listOverall + currentBizList
                break

    driver.quit()
    return listOverall


if __name__ == '__main__':
    import numpy as np
    import glob
    descrp = ['pet hotel', 'pet boarding', 'pet kennel', 'pet sitting', 'pet services']

    dir_path = 'scraped_data/'
    extension = '.csv'
    columns = ['review-name', 'review-rating', 'review-date', 'review-text',
               'url',
               'biz_rating', 'biz_name', 'biz_phone',
               "mon", "tues", "wed", "thurs", "fri", "sat", "sun"]
#    zipcodes_NYC = np.loadtxt('zip_code_SF.txt', delimiter=',')
    zipcodes_NYC = np.loadtxt('zip_code.txt', delimiter=',')
    cities_csv = list(map(int, zipcodes_NYC))

    for des in descrp:
        try:
            des = des.replace(' ', '_')
        finally:
            for i, city in enumerate(cities_csv):
                city = str(city)

                if len(glob.glob(dir_path+city+'_'+des+'_2020*.csv')) == 0:
                    listOverall = startThread(des, city)
                    output = pd.DataFrame.from_records(listOverall, columns=columns)
                    now = time.strftime("%Y%m%d-%H%M%S")

                    output_csv = dir_path + city + '_' + des + '_' + now + extension
                    output.to_csv(output_csv)
