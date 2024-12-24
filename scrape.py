from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver 
from webdriver_manager.chrome import ChromeDriverManager

from file_functions import create_folder, save_json, download_img
from convert_data import convert

import pandas as pd
import time

def wait(func):                 # Wait wrapper: sleeps for 2 seconds after a function is completed to make sure website does not think I am a bot
    '''
    This function is a wrapper to sleep for 2 seconds after the function is called - so that the website does not suspect a bot.

    Args:
        func (function): the function to be wrapped

    Returns:
        wrapper (function): wrapped function
    '''
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        time.sleep(2)
        return output
    return wrapper

def dict_append(dict, dict_to_add):
    if len(dict) == 0:
        for key in dict_to_add:
            if dict_to_add[key][0] == None:
                dict_to_add[key] == ['N/A']
        return dict_to_add
    elif len(dict_to_add) == 0:
        return dict
    else:
        for key in dict:
            if dict_to_add[key][0] == None:
                dict[key] += ['N/A']
            else:
                dict[key] += dict_to_add[key]
        return dict

class Scraper:
    '''
    This class is used to represent a END Clothing scraping tool.

    Attributes:
        driver (selenium.webdriver): a driver to drive the selenium scraping in a browser.
        landing_page (bool): true when the browser is on the landing page.
        img_id (int): id for naming image file names.
        sfirst_search (bool): true if this is the first search and therefore the browser should check for the cookies banner and other banner.
    '''
    # Constructor
    def __init__(self):
        '''
        See help(Scraper) for accurate signature
        '''
        options = Options()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')     
        options.headless = True
        self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        self.landing_page = True
        self.img_id = 0
        self.first_search = True

    # Methods
    @wait
    def look_at(self, url):
        '''
        This function is used to look at a certain url using a selenium driver.

        Args:
            url (str): the url to look at.
        '''
        self.driver.get(url)
        delay = 5
        if self.first_search == True:
            try:
                WebDriverWait(self.driver, delay).until(EC.presence_of_element_located((By.XPATH, '//div[@class="con-wizard"]')))
                accept_cookies_button = WebDriverWait(self.driver, delay).until(EC.presence_of_element_located((By.XPATH, '//button[@class="btn primary"]')))
                accept_cookies_button.click()
                try:
                    WebDriverWait(self.driver, delay).until(EC.presence_of_element_located((By.XPATH, '//button[@aria-label="Close"]')))
                    accept_cookies_button = WebDriverWait(self.driver, delay).until(EC.presence_of_element_located((By.XPATH, '//button[@aria-label="Close"]')))
                    accept_cookies_button.click()
                except TimeoutException:
                    print("Loading took too much time!")

            except TimeoutException:
                print("Loading took too much time!")

            self.first_search = False

    @wait
    def search(self, string):
        '''
        This function is used to search something in the search bar.

        Args:
            string (str): the string to be searched.
        '''
        search_bar = self.driver.find_element(by=By.XPATH, value='//*[@id="yfin-usr-qry"]')
        search_bar.click()
        search_bar.send_keys(string)
        search_bar.send_keys(Keys.RETURN)
        self.landing_page = False

    @wait
    def scroll_bottom(self):
        '''
        This function is used to scroll to the bottom of the page.
        '''
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    @wait
    def scroll(self, pixels):
        '''
        This function is used to scroll by a specific number of pixels.

        Args:
            pixels (int): the number of pixels to scroll down by.
        '''
        self.driver.execute_script(f"window.scrollBy(0,{pixels});")

    def ticker_to_link(self, tickers):
        '''
        This function is used to turn a list of strings containing ticker symbols to a list of corresponding yahoo finance urls.

        Args:
            tickers (list[str]): the list of tickers

        Returns:
            links (): the list of urls
        '''
        links = []
        for ticker in tickers:
            links.append((f'https://finance.yahoo.com/quote/{ticker}/', ticker))
        return links

    def __extract_statistics_page(self):
        '''
        This function is used to extract the data from the statsitics page of an equity.

        Returns:
            data_dict (dictionary): the dictionary of data containing the statistics from the page
        '''
        # Whole table
        table_section = self.driver.find_element(by=By.XPATH, value='//section[@data-yaft-module="tdv2-applet-KeyStatistics"]')
        table_element = table_section.find_elements(by=By.XPATH, value='./div')[-1]
        sub_table_elements = table_element.find_elements(by=By.XPATH, value='./div')

        valuation_measures = sub_table_elements[0]
        valuation_measure_rows = valuation_measures.find_elements(by=By.XPATH, value='.//tr')
        trading_information = sub_table_elements[1]
        trading_information_rows = trading_information.find_elements(by=By.XPATH, value='.//tr')
        financial_highlights = sub_table_elements[2]
        financial_highlight_rows = financial_highlights.find_elements(by=By.XPATH, value='.//tr')

        # Valuation measures
        market_cap = valuation_measure_rows[0].find_elements(by=By.XPATH, value='.//td')[1].text
        market_cap = convert(market_cap)
        trailing_pe = valuation_measure_rows[2].find_elements(by=By.XPATH, value='.//td')[1].text
        trailing_pe = convert(trailing_pe)
        forward_pe = valuation_measure_rows[3].find_elements(by=By.XPATH, value='.//td')[1].text
        forward_pe = convert(forward_pe)
        trailing_ps = valuation_measure_rows[5].find_elements(by=By.XPATH, value='.//td')[1].text
        trailing_ps = convert(trailing_ps)

        # Financial highlights
        profit_margin = financial_highlight_rows[2].find_elements(by=By.XPATH, value='.//td')[1].text
        profit_margin = convert(profit_margin)
        return_on_assets = financial_highlight_rows[4].find_elements(by=By.XPATH, value='.//td')[1].text
        return_on_assets = convert(return_on_assets)
        ebitda = financial_highlight_rows[10].find_elements(by=By.XPATH, value='.//td')[1].text
        ebitda = convert(ebitda)
        current_ratio = financial_highlight_rows[-4].find_elements(by=By.XPATH, value='.//td')[1].text
        current_ratio = convert(current_ratio)

        # Trading information
        short_ratio = trading_information_rows[15].find_elements(by=By.XPATH, value='.//td')[1].text
        short_ratio = convert(short_ratio)
        
        # Collate data into dictionary
        data_dict = {'market_cap': [market_cap], 
                    'trailing_pe': [trailing_pe], 
                    'forward_pe': [forward_pe],
                    'trailing_ps': [trailing_ps],
                    'profit_margin': [profit_margin],
                    'return_on_assets': [return_on_assets],
                    'ebitda': [ebitda],
                    'current_ratio': [current_ratio],
                    'short_ratio': [short_ratio]}

        return data_dict

    def __extract_summary_page(self):
        '''
        This function is used to extract data from the summary page.

        Returns:
            data_dict (dictionary): the dictionary of data from the summary page
        '''
        target_estimate = self.driver.find_element(by=By.XPATH, value='//td[@data-test="ONE_YEAR_TARGET_PRICE-value"]').text
        previous_close = self.driver.find_element(by=By.XPATH, value='//td[@data-test="PREV_CLOSE-value"]').text
        data_dict = {'previous_close': [float(previous_close)], 'target_estimate': [float(target_estimate)]}
        return data_dict

    def __extract_data_ticker(self, link):
        '''
        This function is used to extract the data for the specific ticker.

        Args:
            link (str): the link to the specific ticker

        Returns:
            data (dicitonary): the dictionary of all data for one ticker
        '''
        try:
            # Increment id to ensure each ticker has its own unique id
            self.id += 1

            # Initialise a data dictionary with the specific idea for this ticker
            data = {'id': [self.id], 'ticker': [link[1]]}

            # Visit the tickers yahoo finance page 
            self.look_at(link[0])

            # Extract data from the summary page
            data_summary = self.__extract_summary_page()
            data = data | data_summary

            # Move to the statistics page by clicking on the button
            self.driver.find_element(by=By.XPATH, value='//li[@data-test="STATISTICS"]').click()

            # Wait 1 seconds so that the website does not suspect a bot
            time.sleep(1)

            # Extract data from the statistics page
            data_statistics = self.__extract_statistics_page()
            data = data | data_statistics

            return data
        except:
            return {}

    def extract_all_data(self, tickers):
        '''
        This function is used to extract the data for all tickers and save it to a json file for each ticker.

        Args:
            tickers (list[str]): the list of tickers
        '''
        # Collect a list of links to the tickers to visit
        list_of_links = self.ticker_to_link(tickers)

        # Initialise the ids of the data rows which will then be returned
        self.id = 0

        # Create a raw data folder to store data
        folder_path = create_folder('raw_data')

        # Loop through the list of links and extract the data needed
        data_dictionary = dict()

        for link in list_of_links:
            data = self.__extract_data_ticker(link)
            data_dictionary = dict_append(data_dictionary, data)
            save_json(data, link[1], folder_path)    # Save file in raw_data folder

        self.all_data_dict = data_dictionary
        save_json(data_dictionary, 'all_data', folder_path)

        return data_dictionary

    def extract_all_data_letter(self):
        '''
        This function is used to extract the data for all tickers on the nasdaq starting with letter and save it to a json file for each ticker.

        Args:
            letter (str): the letter for ticker to start with
        '''
        letter = input('Please enter a single letter: ')

        if letter.isalpha() and len(letter) == 1:
            
            df_nasdaq = pd.read_csv('nasdaq_tickers.csv').dropna(axis=0)
            tickers = [tick for tick in df_nasdaq['Symbol'].to_list() if tick[0] == letter.upper()]

            # Collect a list of links to the tickers to visit
            list_of_links = self.ticker_to_link(tickers)

            # Initialise the ids of the data rows which will then be returned
            self.id = 0

            # Create a raw data folder to store data
            folder_path = create_folder('raw_data')

            # Loop through the list of links and extract the data needed
            data_dictionary = dict()

            for link in list_of_links:
                data = self.__extract_data_ticker(link)
                data_dictionary = dict_append(data_dictionary, data)
                save_json(data, link[1], folder_path)    # Save file in raw_data folder

            self.all_data_dict = data_dictionary
            save_json(data_dictionary, 'all_data', folder_path)

            return data_dictionary
        else:
            self.extract_all_data_letter()

    def extract_logo(self):
        '''
        This function is used to extract the yahoo finance logo and download it.
        '''
        logo_element = self.driver.find_element(by=By.XPATH, value='//a[@id="header-logo"]')
        link_long = logo_element.get_attribute('style')
        index_start = link_long.find('url(')
        link_img = link_long[index_start+4:-2]
        self.img_id += 1
        download_img(link_img, self.img_id)

    def end_session(self):
        '''
        This function is used to end the session.
        '''
        self.driver.quit()

if __name__ == "__main__":
    
    yahoo_finance = Scraper()
    data = yahoo_finance.extract_all_data_letter()
    df = pd.DataFrame.from_dict(data)
    print(df.head())
    yahoo_finance.end_session()