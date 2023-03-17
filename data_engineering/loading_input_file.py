import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_data_from_kaggle():
    """
    This function download data from kaggle source and creates new file.

    :return: None
    """
    api = KaggleApi()

    # API token has been provided through environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
    api.authenticate()

    dataset = 'kmader/skin-cancer-mnist-ham10000'
    file_name = 'HAM10000_metadata.csv'
    file_path = '../data/input/'
    dataset_dir = file_path + file_name
    new_file_name = file_path + 'input_data_from_kaggle.csv'

    # Checking whether file already is downloaded
    if os.path.exists(new_file_name):
        print("Found dataset directory, exiting")
        exit(0)

    print("Dataset not found, using kaggle-api tool for download")

    # Downloading file from kaggle (link: 'https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000') and
    # creating new file
    api.dataset_download_files(dataset=dataset,
                               path='../data/input',
                               unzip=True)

    # Renaming downloaded file
    os.rename(dataset_dir, new_file_name)


def main():
    download_data_from_kaggle()


if __name__ == '__main__':
    main()
