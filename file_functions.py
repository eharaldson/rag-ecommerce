import os
import json
import requests
from datetime import datetime

def create_folder(folder_name):
    '''
    This function is used to create a folder in the current directory

    Args:
        folder_name (str): folder name to be created.

    Returns:
        final_directory: the directory of the folder.
    '''
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, folder_name)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    return final_directory

def save_json(dict, ticker, file_path):
    '''
    This function is used to save a dictionary to a file

    Args:
        dict (dictionary): the dictionary to be saved.
        file_path (str): the path for the file to be saved.
    '''
    completeName = os.path.join(file_path, ticker+".json")  
    if not os.path.exists(completeName):
        with open(completeName, "w") as file:
            json.dump(dict, file)     

def download_img(img_url, id):  # Download an image with correct file name in the images folder of raw_data
    '''
    This function is used to download an image from a url to a images folder.

    Args:
        img_url (str): the image url of the image to be downloaded
        id (int): the id of the image to be saved
    '''
    img_data = requests.get(img_url[1:-1]).content
    date_string = datetime.today().strftime('%d-%m-%Y %H:%M:%S')
    date = date_string[:10].replace('-', '')
    time = date_string[11:].replace(':', '')
    directory = create_folder('raw_data/images')
    file_name = os.path.join(directory, date+'_'+time+'_'+str(id)+'.png')
    with open(file_name, "wb") as file:
        file.write(img_data)