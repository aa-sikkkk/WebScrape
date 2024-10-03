#V1.1
# import required modules 

import json 
import requests
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from beautifultable import BeautifulTable



def load_json(database_json_file="scraped_data.json"):
    """
    This function will load json data from scraped_data.json
    if file exist it will load data from it else creates an empty one"""
    try:
        with open(database_json_file, "r") as read_it: 
            all_data_base = json.loads(read_it.read())
            return all_data_base
    except:
        all_data_base = dict()
        return all_data_base


def save_scraped_data_in_json(data, database_json_file="scraped_data.json"):
    """
        This function Save the scraped data in json format. scraped_data.json file if it exist else create it.
        if file already exist you can view previous scraped data in it.
    """
    file_obj =  open(database_json_file, "w")
    file_obj.write(json.dumps(data))
    file_obj.close()


def existing_scraped_data_init(json_db):
    """
    This function will check if scraped_data key exist in json_db or not
    """
    scraped_data = json_db.get("scraped_data")
    if scraped_data is None:
        json_db['scraped_data'] = dict()

    return None


def scraped_time_is():
    """
    This function will return current date and time in string format
    """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string

def process_url_request(website_url):
    """
    This function will process the url request and return the soup object
    """
    requets_data = requests.get(website_url)
    if requets_data.status_code == 200:
        soup = BeautifulSoup(requets_data.text, 'lxml')
        return soup
    return None

def proccess_beautiful_soup_data(soup):
    return {
        'title': soup.find('title').text if soup.find('title') else 'No title found',
        'all_anchor_href': [i['href'] for i in soup.find_all('a', href=True)],
        'all_anchors': [str(i) for i in soup.find_all('a')],
        'all_images_data': [str(i) for i in soup.find_all('img')],
        'all_images_source_data': [i.get('src') for i in soup.find_all('img')],
        'all_h1_data': [i.text for i in soup.find_all('h1')],
        'all_h2_data': [i.text for i in soup.find_all('h2')],
        'all_h3_data': [i.text for i in soup.find_all('h3')],
        'all_p_data': [i.text for i in soup.find_all('p')]
    }




# Main program starts from here
while True:

    print("""  ================ Welcome to this scraping program =============
    ==>> press 1 for checking existing scraped websites
    ==>> press 2 for scrap a single website
    ==>> press 3 for exit
    """)

    choice = int(input("==>> Please enter your choice :"))

    # load data
    local_json_db = load_json()
    existing_scraped_data_init(local_json_db)

    if choice == 1:
        # Using Beautiful table for presenting scraped data in a good way !!
        # Read more about from this link https://beautifultable.readthedocs.io/en/latest/index.html
        scraped_websites_table = BeautifulTable()
        scraped_websites_table.columns.header = ["Sr no.", "Allias name ", "Website domain", "title",   "Scraped at", "Status"]
        scraped_websites_table.set_style(BeautifulTable.STYLE_BOX_DOUBLED)
        

        local_json_db = load_json()
        for count,  data in enumerate(local_json_db['scraped_data']):
           scraped_websites_table.rows.append([count + 1, 
                            local_json_db['scraped_data'][data]['alias'], 
                            local_json_db['scraped_data'][data]['domain'], 
                            local_json_db['scraped_data'][data]['title'], 
                            local_json_db['scraped_data'][data]['scraped_at'], 
                            local_json_db['scraped_data'][data]['status']])
        # all_scraped_websites = [websites['name'] for websites in local_json_db['scraped_data']]
        if not local_json_db['scraped_data']:
            print('===> No existing data found !!!')
        print(scraped_websites_table)
    
    elif choice == 2:
        print()
        url_for_scrap = input("===> Please enter url you want to scrap:")
        is_accessable = process_url_request(url_for_scrap)
        if is_accessable:
            scraped_data_packet = proccess_beautiful_soup_data(is_accessable)
            print()
            print(' =====> Data scraped successfully !!!')
            key_for_storing_data = input("enter alias name for saving scraped data :")
            scraped_data_packet['url'] = url_for_scrap
            scraped_data_packet['name'] = key_for_storing_data
            scraped_data_packet['scraped_at'] = scraped_time_is()
            if key_for_storing_data in  local_json_db['scraped_data']:
                key_for_storing_data = key_for_storing_data + str(scraped_time_is())
                print("Provided key is already exist so data stored as : {}".format(key_for_storing_data))
            scraped_data_packet['alias'] = key_for_storing_data
            scraped_data_packet['status'] = True
            scraped_data_packet['domain'] = urlparse(url_for_scrap).netloc

            local_json_db['scraped_data'][key_for_storing_data] = scraped_data_packet
            print(
                'scraped data is:', local_json_db['scraped_data'][key_for_storing_data]
            )
            save_scraped_data_in_json(local_json_db)
            # load data
            local_json_db = load_json()
            print(' =====> Data saved successfully !!!')
            print()
    elif choice == 3:
        print('Thank you for using !!!')
        break

    elif choice == 4:
        print('Thank you for using !!!')
        break

    else:
        print("enter a valid choice ")