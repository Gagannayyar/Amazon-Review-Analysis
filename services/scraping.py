
import requests
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import datetime
import time
from random import choice
import os

HEADERS = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/90.0.4430.212 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})

#Getting the proxy list
working_directory = os.getcwd()
wp = pd.read_csv('Working_Proxies.csv')
proxy_list = list(wp['http'])




class ExtractReviews:
    
    def __init__() -> None:
        pass

    @staticmethod
    def html_code(url,proxy):
        proxy= {'http': choice(proxy_list)}
        htmldata = requests.get(url,proxies=proxy,headers=HEADERS).text
        soup = BeautifulSoup(htmldata, 'html.parser')
        return (soup)

    def get_product_name(soup_object) -> str:
        product_name = soup_object.find(id='productTitle').get_text()
        return str(product_name).strip()

    def get_product_category(soup_object) -> list:
        cat_list = []
        category = soup_object.find_all('span', class_='a-list-item')
        for items in category:
            cat_list.append(items.get_text().strip())
        
        return cat_list[:2]

    def get_total_pages_reviews(soup_object) -> int:
        page_number = soup_object.find('span',id ='acrCustomerReviewText').text.strip(' ratings')
        #Convert to integer
        page_number = int("".join(page_number.split(',')))
        total_pages = int(page_number/10)
        if total_pages > 1000:
            total_pages = 1000
        return total_pages

    def get_reviews_body(url,pages: int):
        #Getting the url ready for reviews
        review_list = []
        url_amend = []
        url = url.replace('dp','product-reviews')
        url_amend = url.split('/')
        url_amend.pop()
    #Starting the loop to get all the reviews in a dataframe
        for i in range(1, pages):
            page_num = f"ref=cm_cr_arp_d_paging_btm_next_{i}?ie=UTF8&reviewerType=all_reviews&pageNumber={i}"
            proxy= {'http': choice(proxy_list)}
            url_amend.append(page_num)
            url_new = '/'.join(url_amend)
            soup_reviews = ExtractReviews.html_code(url_new,proxy)
            reviews = soup_reviews.find_all("div",{'data-hook': 'review'})
            print("Extracting Reviews.......")
            for items in reviews:
                review_list.append(items.find('span',{'data-hook': 'review-body'}).text.strip())
            url_amend = url.split('/')
            url_amend.pop()
            time.sleep(5)
        
        return review_list

    def list_to_dataframe(list):    
        df = pd.DataFrame({'comments': list})
        return df