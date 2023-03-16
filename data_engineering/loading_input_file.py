import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()

# API token has been provided through environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
api.authenticate()

dataset = 'kmader/skin-cancer-mnist-ham10000'
file_name = 'HAM10000_metadata.csv'
file_path = '../data/input/'
dataset_dir = file_path + file_name

# Checking whether file already is downloaded
if os.path.exists(dataset_dir):
    print("Found dataset directory, exiting")
    exit(0)

print("Dataset not found, using kaggle-api tool for download")

# Downloading file from kaggle (link: 'https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000')
api.dataset_download_file(dataset=dataset,
                          file_name=file_name,
                          path='../data/input')
