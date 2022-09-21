from secrets import randbelow
import requests
from bs4 import BeautifulSoup as bs
from random import choice
import pandas as pd
import os


class Proxies:

    def __init__() -> None:
        pass

    def get_proxy_list(proxy_url:str):
        request = requests.get(proxy_url)
        soup = bs(request.content, "html.parser").find_all("td", {
                "class":"blob-code blob-code-inner js-file-line"
                })
        proxies = [proxy.text for proxy in soup]
        return proxies

    def random_proxy(proxy_list:list):
        return{"https": choice(proxy_list)}

    def get_working_proxies(proxy_list):
        """
        Checking the working proxies from a list of proxies
        Param:
            proxy_list: A list of proxies to be checked
        """
        working_proxies = []
        for proxy in proxy_list:
            proxy = {"http": proxy}
            print(proxy)
            try:
                #check every proxy and append the one with 200 status code to the list
                request = requests.get("https://www.google.com",proxies=proxy,timeout=2)
                print(request.status_code)
                if request.status_code == 200:
                    working_proxies.append(proxy)
            except:
                pass
        return working_proxies

def file_exist(file):
    import os.path
    file_exists = os.path.exists(f"{working_directory}/Working_Proxies")
    if file_exists == True:
        os.remove(f"{working_directory}/Working_Proxies")
        file.to_csv(f"{working_directory}/Working_Proxies.csv")
    else:
        file.to_csv(f"{working_directory}/Working_Proxies.csv")


                
proxy_url = "https://github.com/clarketm/proxy-list/blob/master/proxy-list-raw.txt"
raw_proxy_list = Proxies.get_proxy_list(proxy_url=proxy_url)
working_proxies = Proxies.get_working_proxies(raw_proxy_list)
working_proxiesDF = pd.DataFrame(working_proxies)
working_directory = os.getcwd()
file_exist(working_proxiesDF)
print(working_proxies)